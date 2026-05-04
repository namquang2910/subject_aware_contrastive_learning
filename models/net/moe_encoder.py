import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from models.net.CNNEncoder import CNNEncoder
from models.net.utils import get_conv1d_output_dim, get_maxpool1d_output_dim

class StemEncoder(nn.Module):
    def __init__(self,input_dim: int, 
                 dropout_prob:int = 0.1, 
                 kernel_size: int = 7, 
                 stride:int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = self.last_dim = 256
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout_prob = dropout_prob
        self.cnn_output_dim = self._get_cnn_output_dim()
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=self.kernel_size, stride=self.stride),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=self.dropout_prob),

            nn.Conv1d(32, 64, self.kernel_size, self.stride),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=self.dropout_prob),

            nn.Conv1d(64, 128, self.kernel_size, self.stride),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=self.dropout_prob),
            
            nn.Conv1d(128, 256, self.kernel_size, self.stride),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=self.dropout_prob),
        )
        
        self.gap = nn.AdaptiveAvgPool1d(1)

    def _get_cnn_output_dim(self):
        out1 = get_conv1d_output_dim(self.input_dim, 0, 1, self.kernel_size, self.stride)
        out1b = get_maxpool1d_output_dim(out1, 0, 1, 2, 2)
        out2 = get_conv1d_output_dim(out1b, 0, 1, self.kernel_size, self.stride)
        out2b = get_maxpool1d_output_dim(out2, 0, 1, 2, 2)
        out3 = get_conv1d_output_dim(out2b, 0, 1, self.kernel_size, self.stride)
        out3b = get_maxpool1d_output_dim(out3, 0, 1, 2, 2)
        out4 = get_conv1d_output_dim(out3b, 0, 1, self.kernel_size, self.stride)
        out4b = get_maxpool1d_output_dim(out4, 0, 1, 2, 2)
        final_out = 256 * out4b  # number of output channels * output dim of last layer
        return final_out

    def forward(self, x, return_embedding = False):
        if len(x.shape) == 2:
            x = torch.reshape(x, (x.shape[0], 1, -1))
        z = self.cnn_layers(x)
        h = self.gap(z).squeeze(-1)
        z = torch.reshape(z, (z.shape[0], -1))
        if return_embedding:
            return h, z
        return h
    
    
class MoEDualBranchEncoder(nn.Module):
    def __init__(self, input_dim: int, dropout_prob: float = 0.1, kernel_size: int = 7,
                 stride: int = 1, output_dim: int = 64, projection_output: int = 32,
                 use_bn: bool = True):
        super().__init__()
        self.use_bn = use_bn
        #self.stem   = CNNEncoder(input_dim, dropout_prob, kernel_size, stride)
        self.stem = StemEncoder(input_dim, dropout_prob, kernel_size, stride)
        D = self.stem.output_dim
 
        self.proj_inv = nn.Sequential(
            nn.Linear(D, output_dim), self._bn(output_dim), nn.ReLU(),
            nn.Linear(output_dim, projection_output), self._bn(projection_output),
        )
        self.proj_spec = nn.Sequential(
            nn.Linear(D, output_dim), self._bn(output_dim), nn.ReLU(),
            nn.Linear(output_dim, projection_output), self._bn(projection_output),
        )

        self.output_dim        = D
        self.cnn_output_dim    = self.stem.cnn_output_dim
        self.projection_output = projection_output

    def _bn(self, dim):
        return nn.BatchNorm1d(dim) if self.use_bn else nn.Identity()

    def forward(self, x, return_embeddings: bool = False):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h, z      = self.stem(x, return_embedding=True)
        z_inv  = self.proj_inv(h)
        z_spec = self.proj_spec(h)
        h_out  = torch.cat([z_inv, z_spec], dim=1)
        if return_embeddings:
            return h, h_out,z, z_inv, z_spec
        return h_out, z_inv, z_spec