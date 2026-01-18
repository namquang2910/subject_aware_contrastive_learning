import torch
import torch.nn as nn

import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dropout=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(in_channel)
        self.elu1 = nn.ELU()
        self.Conv1 = nn.Conv1d(in_channel, out_channel, kernel_size,
                               padding=kernel_size//2)
        self.drop1 = nn.Dropout(p=dropout)

        self.norm2 = nn.BatchNorm1d(out_channel)
        self.elu2 = nn.ELU()
        self.Conv2 = nn.Conv1d(out_channel, out_channel, kernel_size,
                               padding=kernel_size//2)
        self.drop2 = nn.Dropout(p=dropout)

        if in_channel != out_channel:
            self.shortcut = nn.Conv1d(in_channel, out_channel, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
   
    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.Conv1(self.elu1(self.norm1(x)))
        x = self.drop1(x)

        x = self.Conv2(self.elu2(self.norm2(x)))
        x = self.drop2(x)

        return x + shortcut

    
class ResNet25(nn.Module):
    """
    Input:  (B, 2, 704) by default
    Output: embeddings (B, embed_dim) or logits if num_classes is set
    """
    def __init__(self, in_channels: int = 1, embed_dim: int = 64, output_dim: int | None = None, dropout=0.2):
        super().__init__()
        self.output_dim = output_dim
        # Stem
        self.stem = nn.Conv1d(in_channels, 32, kernel_size=13, padding=13 // 2, bias=False)

        # Encoder G (per your diagram)
        self.block1 = ResBlock(32, 32, kernel_size=11, dropout=dropout)
        self.pool1  = nn.MaxPool1d(kernel_size=4, stride=4)

        self.block2 = ResBlock(32, 64, kernel_size=9, dropout=dropout   )
        self.pool2  = nn.MaxPool1d(kernel_size=4, stride=4)

        self.block3 = ResBlock(64, 128, kernel_size=7, dropout=dropout)
        self.pool3  = nn.MaxPool1d(kernel_size=2, stride=2)

        self.block4 = ResBlock(128, 256, kernel_size=7, dropout=dropout )

        self.enc_act = nn.ELU(inplace=True)
        self.gap = nn.AdaptiveAvgPool1d(1)   # -> (B,256,1)

        # Model F head (MLP 256->128->128->embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.ReLU())

    def forward(self, x, return_embedding: bool = False):
        # Encoder
        x = self.stem(x)
        x = self.block1(x); x = self.pool1(x)
        x = self.block2(x); x = self.pool2(x)
        x = self.block3(x); x = self.pool3(x)
        x = self.block4(x)
        x = self.enc_act(x)
        z = self.gap(x).squeeze(-1)  # (B,256)

        if return_embedding:
            return z                 # embedding only
        return self.fc(z)    # logits

