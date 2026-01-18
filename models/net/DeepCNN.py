"""
Class for basic CNN encoder.
"""

import torch.nn as nn
import torch
from models.net.utils import get_conv1d_output_dim, get_maxpool1d_output_dim

class DeepCNN(nn.Module):
    """
    CNN-based encoder.
    """
    def __init__(self, input_dim, dropout_prob=0.1, kernel_size=7, stride=3, output_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        self.kernel_size = kernel_size
        self.stride = stride
        self.cnn_layers = self._get_cnn_layers()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc =  nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = x.float() 
        if len(x.shape) == 1:
            # [L] -> [1, 1, L]
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 2:
            # [B, L] -> [B, 1, L]
            x = x.unsqueeze(1)
        h = self.cnn_layers(x)
        # reshape so that representations are 1D
        h = self.gap(h).squeeze(-1) 
        # apply fully connected layer
        h = self.fc(h)
        return h

    def _get_cnn_layers(self):
        dropout = self.dropout_prob

        cnn_layers = nn.Sequential(
            # --- Block 1 ---
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=32, stride=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=32, stride=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            # --- Block 2 ---
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16, stride=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=16, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            # --- Block 3 ---
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=12, stride=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=12, stride=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            # --- Block 4 ---
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=8, stride=2),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=8, stride=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        return cnn_layers
