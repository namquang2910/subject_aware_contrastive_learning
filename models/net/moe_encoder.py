import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class StemEncoder(nn.Module):
    def __init__(self,input_dim: int, 
                 dropout_prob:int = 0.1, 
                 kernel_size: int = 7, 
                 stride:int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 256
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(in_channels = 1, out_channels= 32, kernel_size = kernel_size, stride = stride),
            nn.BatchNorm1d(num_features= 32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 2),
            nn.Dropout(p= dropout_prob),

            nn.Conv1d(in_channels = 32, out_channels= 64, kernel_size = kernel_size, stride = stride),
            nn.BatchNorm1d(num_features= 64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 2),
            nn.Dropout(p= dropout_prob),

            nn.Conv1d(in_channels = 64, out_channels= 128, kernel_size = kernel_size, stride = stride),
            nn.BatchNorm1d(num_features= 128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 2),
            nn.Dropout(p= dropout_prob),

            nn.Conv1d(in_channels = 128, out_channels= 256, kernel_size = kernel_size, stride = stride),
            nn.BatchNorm1d(num_features= 256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 2),
            nn.Dropout(p= dropout_prob),
        )
        
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        if len(x.shape) == 2:
            x = torch.reshape(x, (x.shape[0], 1, -1))
        h = self.cnn_layers(x)
        h = self.gap(h).squeeze(-1)
        
        return h
    

class MoEDualBranchEncoder(nn.Module):
    """
    forward(x) returns:
        h_out  [B, P]   router-weighted combination of expert outputs
        z_inv  [B, P]   invariant expert (proj_inv) output
        z_spec [B, P]   subject-specific expert (proj_spec) output
        g      [B, 2]   routing weights [g_inv, g_spec]
    """
    def __init__(
        self,
        input_dim: int, 
        dropout_prob:int = 0.1, 
        kernel_size: int = 7, 
        stride:int = 1,
        output_dim: int = 64,
        projection_output:     int   = 32,
        router_temperature: float = 1.0,
    ):
        super().__init__()


        #  Shared stem 
        self.stem = StemEncoder(input_dim, dropout_prob, kernel_size, stride)

        #  Expert 1: Invariant projection head 
        self.proj_inv = nn.Sequential(
            nn.Linear(self.stem.output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, projection_output),
            nn.BatchNorm1d(projection_output),
        )

        #  Expert 2: Subject-Specific projection head 
        self.proj_spec = nn.Sequential(
            nn.Linear(self.stem.output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, projection_output),
            nn.BatchNorm1d(projection_output),
        )
        

        self.output_dim        = output_dim
        self.projection_output = projection_output


    def forward(self, x):
        """x: [B, 1, L] or [B, L]"""
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # ── 1. Shared stem ────────────────────────────────────────
        h = self.stem(x)                        # [B, D]

        # ── 2. Both expert heads run on the same h ────────────────
        z_inv  = self.proj_inv(h)               # [B, P]
        z_spec = self.proj_spec(h)              # [B, P]

        # ── 3. Router-weighted combination ────────────────────────
        #g = self.g                              # scalar in (0, 1)
        #h_out = g * z_inv + (1 - g) * z_spec   # [B, P]
        h_out = torch.cat([z_inv, z_spec], dim=-1)
        return h_out, z_inv, z_spec