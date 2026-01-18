import copy
import torch.nn as nn
import torch
from models.model import Model
from models.utils import get_base_encoder
from torch.autograd import Function

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

class AdversarialContrastiveModel(Model):
    """
    Contrastive model with adversarial subject classifier
    to encourage subject-invariant representations.
    """
    def __init__(self, base_encoder, projection_output=32,
                 num_subjects=None, grl_lambda=1.0, device=None):
        super().__init__(device=device)
        assert num_subjects is not None, "num_subjects must be specified for adversarial subject head."

        self.loss_fn = None  # contrastive loss, same as before (NCELoss, etc.)
        self.projection_output = projection_output
        self.grl_lambda = grl_lambda

        # encoder
        self.encoder = copy.deepcopy(base_encoder)

        embedding_dim = self.encoder.cnn_out_dim
        dim_mlp = self.encoder.output_dim

        # projection head for contrastive loss
        if self.projection_output is not None:
            self.projection_head = nn.Sequential(
                                nn.Linear(dim_mlp, self.projection_output),
                                nn.BatchNorm1d(self.projection_output),
                                )

        # subject classifier head – simple MLP
        self.subject_head = nn.Sequential(
            nn.Linear(embedding_dim, dim_mlp*4),
            nn.ReLU(inplace=True),
            nn.Linear(dim_mlp*4, dim_mlp*2),
            nn.ReLU(inplace=True),
            nn.Linear(dim_mlp*2, num_subjects)
        )
        self.subject_criterion = nn.CrossEntropyLoss()

    def forward(self, batch, return_loss = False):
        """
        view1, view2: dicts with 'x'
        subject_ids: tensor [B] with subject indices
        """
        self._check_loss_fn()

        x1 = batch['x1']['x']
        x2 = batch['x2']['x']

        if x1.dim() == 2:
            x1 = x1.unsqueeze(1)
            x2 = x2.unsqueeze(1)

        x1 = x1.to(self.device, non_blocking=True).float()
        x2 = x2.to(self.device, non_blocking=True).float()
        subject_ids = batch['subject_id_int'].to(self.device, non_blocking=True).long()

        # ----- encoder -----
        h1, z1 = self.encoder(x1, return_embedding=True)   # [B, D]
        h2, z2 = self.encoder(x2, return_embedding=True)   # [B, D]

        # ----- contrastive branch -----
        h1 = self.projection_head(h1)  # [B, P]
        h2 = self.projection_head(h2)  # [B, P]
        contrastive_loss = self.loss_fn(h1, h2)

        # ----- adversarial subject branch (on h1 or concat) -----
        # you could also use torch.cat([h1, h2], dim=0) and repeat labels

        #h_rev = grad_reverse(z1, self.grl_lambda)
        subj_logits = self.subject_head(z1)  # [B, num_subjects]
        probs = torch.softmax(subj_logits, dim=1)
        subject_loss = self.subject_criterion(subj_logits, subject_ids)
        true_subject_probs = probs[torch.arange(probs.size(0)), subject_ids]
        eps = 1e-8
        clamped = torch.clamp(1 - true_subject_probs, min=eps, max=1.0)
        adversarial_loss = -torch.log(clamped).mean()

        total_loss = contrastive_loss + self.grl_lambda * adversarial_loss  # GRL makes this adversarial for the encoder
        if return_loss:
            return total_loss, adversarial_loss, contrastive_loss, contrastive_loss
        return total_loss, adversarial_loss
