"""
Basic architecture for contrastive learning network.
"""
import copy
import torch.nn as nn
import torch
from models.model import Model
from models.utils import get_base_encoder
#from loss.soft_dtw_cuda import SoftDTW
class ContrastiveModel(Model):
    """
    Class for basic contrastive learning model.
    """
    def __init__(self, base_encoder, projection_output=32, device= None, use_softdtw = False):
        """
        encoder_name: name of network to use for mapping from raw data to learned representations (h)
        encoder_args: arguments for encoder network
        encoder_state_dict (optional): path to Pytorch state dict that contains weights to initialize model with
        transform_head_name (optional): name of network for mapping from learned representations (h) to metric embedding
        transform_head_args (optional): arguments for transform head network
        transform_head_state_dict (optional): path to Pytorch state dict that contains weights to initialize model with
        """
        super().__init__(device=device)
        self.loss_fn = None
        self.subject_level = True
        self.use_softdtw = use_softdtw
        self.projection_output = projection_output
        self.encoder = copy.deepcopy(base_encoder)
        
        dim_mlp = self.encoder.output_dim
        # separate heads for query and key branches
        self.projection_head = nn.Sequential(
                                nn.Linear(dim_mlp, self.projection_output),
                                nn.BatchNorm1d(self.projection_output),
                                )

    def forward(self, batch, return_loss=True):
        self._check_loss_fn()
        x1 = batch['x1']['x']
        x2 = batch['x2']['x']
        if x1.dim() == 2:
            # [B, L] -> [B, 1, L]
            x1 = x1.unsqueeze(1)
            x2 = x2.unsqueeze(1)
        # Prepare the data
        x1 = x1.to(self.device, non_blocking=True).float()
        x2 = x2.to(self.device, non_blocking=True).float()
        

        q_embed = self.encoder(x1)
        k_embed = self.encoder(x2)

        q_embed = self.projection_head(q_embed)
        k_embed = self.projection_head(k_embed)

        loss = self.loss_fn(q_embed, k_embed)
        return loss, loss, loss