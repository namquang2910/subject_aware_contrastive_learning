import copy
import torch
import torch.nn as nn
from torch.autograd import Function
from models.model import Model
from loss.cl_loss import NCELoss
from torch.nn import functional as F
# -------- Gradient Reversal --------
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


# -------- Main Model --------
class SubjectAwareContrastiveModel(Model):
    """
    z_base : instance-level contrastive (SimCLR)
    z_sub  : subject-level contrastive (subject-aware)
    z_inv  : subject-invariant (adversarial GRL)
    """

    def __init__(
        self,
        base_encoder,
        projection_output=128,
        num_subjects=None,
        grl_lambda=1.0,
        device=None,
    ):
        super().__init__(device=device)
        assert num_subjects is not None

        self.encoder = copy.deepcopy(base_encoder)
        self.grl_lambda = grl_lambda

        dim_mlp = self.encoder.output_dim

        # ---- projections ----
        self.proj_sub  = nn.Sequential(
            nn.Linear(dim_mlp, projection_output),
            nn.BatchNorm1d(projection_output),
            nn.ReLU(),
            nn.Linear(projection_output, projection_output)
        )
        self.proj_inv  = nn.Sequential(
            nn.Linear(dim_mlp, projection_output),
            nn.BatchNorm1d(projection_output),
            nn.ReLU(),
            nn.Linear(projection_output, projection_output),
        )
        self.proj_distill = nn.Sequential(
            nn.Linear(dim_mlp, projection_output),
            nn.BatchNorm1d(projection_output),
            nn.ReLU(),
            nn.Linear(projection_output, projection_output),
        )
        self.distill_projection = nn.Sequential(
            nn.Linear(projection_output * 2, projection_output * 2),
            nn.BatchNorm1d(projection_output * 2),
            nn.ReLU(),
            nn.Linear(projection_output * 2, projection_output)
        )
        
        # placeholders (set externally)
        self.loss_fn_sub  = NCELoss(temperature=0.1)
        self.loss_fn_inv = NCELoss(temperature=0.1)

    def decorrelation_loss(self, z1, z2):
        """
        Barlow Twins-style decorrelation loss
        Encourages z1 and z2 to be decorrelated across features
        """
        # Normalize
        z1 = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-6)
        z2 = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-6)
        
        # Cross-correlation matrix
        batch_size = z1.size(0)
        c = (z1.T @ z2) / batch_size  # [projection_output, projection_output]
        
        # Loss: off-diagonal elements should be zero
        # We want c to be close to zero matrix (since z1 and z2 should be independent)
        off_diagonal = c.pow(2).sum() - c.diagonal().pow(2).sum()
        
        return off_diagonal
    def forward(self, batch, return_loss=False):
        self._check_loss_fn()

        # -------- inputs --------
        x1 = batch["x1"]["x"].to(self.device).float()
        x2 = batch["x2"]["x"].to(self.device).float()
        subject_ids = batch["subject_id_int"].to(self.device).long()

        if x1.dim() == 2:
            x1 = x1.unsqueeze(1)
            x2 = x2.unsqueeze(1)

        # -------- encoder --------
        h1, z1 = self.encoder(x1, return_embedding=True)
        h2, z2 = self.encoder(x2, return_embedding=True)

        # -------- projections --------
        z_sub_1  = self.proj_sub(h1)
        z_sub_2  = self.proj_sub(h2)

        z_inv_1    = self.proj_inv(h1)
        z_inv_2    = self.proj_inv(h2)

        z_distill_1 = self.proj_distill(h1)
        z_distill_2 = self.proj_distill(h2)

        # -------- losses --------
        loss_distill_contrast = self.loss_fn_inv(z_distill_1, z_distill_2)
        loss_sub = self.loss_fn_sub(z_sub_1, z_sub_2, subject_ids) #subject-level contrastive loss
        loss_inv = self.loss_fn_inv(z_inv_1, z_inv_2) #instance-level contrastive loss

        #Adversarial subject classification loss
        #z_inv_rev = grad_reverse(z_inv_1, self.grl_lambda) 
        #subj_logits = self.subject_head(z_inv_rev)
        #loss_classifier = self.subject_criterion(subj_logits, subject_ids)

        # Orthogonal loss to encourage z_sub and z_inv to be different
        loss_dis = (self.decorrelation_loss(z_sub_1, z_inv_1) + 
                    self.decorrelation_loss(z_sub_2, z_inv_2)) / 2
        #distillation loss to align the projections
        z_combined_1 = torch.cat([z_sub_1.detach(), z_inv_1.detach()], dim=1)
        z_combined_2 = torch.cat([z_sub_2.detach(), z_inv_2.detach()], dim=1)

        loss_distill_1 = nn.MSELoss()(z_distill_1, self.distill_projection(z_combined_1))
        loss_distill_2 = nn.MSELoss()(z_distill_2, self.distill_projection(z_combined_2))
        loss_distill = (loss_distill_1 + loss_distill_2) / 2

        #Total loss
        total_loss = loss_sub + loss_inv + loss_distill_contrast + loss_dis * 1.0 + loss_distill * 1.0
        return total_loss,loss_distill_contrast,loss_sub + loss_inv, loss_distill + loss_dis
