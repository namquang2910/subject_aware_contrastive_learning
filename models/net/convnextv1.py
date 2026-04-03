import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, L)
    (2) DwConv -> Permute to (N, L, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3,groups=dim) # depthwise conv
        self.norm = nn.BatchNorm1d(num_features=dim, eps=1e-6)
        self.pwconv1 = nn.Conv1d(dim, 4*dim, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.pwconv2 = nn.Conv1d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma.view(1, -1, 1) * x
        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    def __init__(self, in_chans=1, 
                depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.1, 
                layer_scale_init_value=1e-6,
                ):
        super().__init__()
        self.output_dim = dims[-1]
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv1d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(num_features=dims[0]),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    nn.BatchNorm1d(num_features=dims[i], eps=1e-6),
                    nn.Conv1d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.BatchNorm1d(num_features=dims[-1])
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:                # guard against bias=False layers
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
           # print(x.shape)
            x = self.downsample_layers[i](x)
          #  print(x.shape)
            x = self.stages[i](x)
        return self.norm(x.mean([-1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
      #  print(f"Input shape: {x.shape}")
        x = self.forward_features(x)
        return x


def convnext_stiny(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[1, 1, 3, 1], dims=[32, 64, 128, 256], **kwargs)
    return model

def convnext_tiny(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[64, 128, 256, 512], **kwargs)
    return model

def convnext_small(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    return model