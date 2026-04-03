import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
#  Shared stem  (identical to original StemEncoder)
# ──────────────────────────────────────────────────────────────────────────────
class StemEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        dropout_prob: float = 0.1,
        kernel_size: int = 7,
        stride: int = 1,
    ):
        super().__init__()
        self.input_dim  = input_dim
        self.output_dim = 256
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(1,   32,  kernel_size, stride), nn.BatchNorm1d(32),  nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout_prob),
            nn.Conv1d(32,  64,  kernel_size, stride), nn.BatchNorm1d(64),  nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout_prob),
            nn.Conv1d(64,  128, kernel_size, stride), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout_prob),
            nn.Conv1d(128, 256, kernel_size, stride), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout_prob),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)           # [B, L] -> [B, 1, L]
        h = self.cnn_layers(x)
        return self.gap(h).squeeze(-1)   # [B, 256]


# ──────────────────────────────────────────────────────────────────────────────
#  Router
# ──────────────────────────────────────────────────────────────────────────────
class Router(nn.Module):
    """
    Input  : h  [B, D]
    Output : g  [B, N]   softmax weights, one per expert
    """
    def __init__(self, embed_dim: int, num_experts: int, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, num_experts),
        )

    def forward(self, h):
        logits = self.gate(h)                               # [B, N]
        return F.softmax(logits / self.temperature, dim=-1) # [B, N]


# ──────────────────────────────────────────────────────────────────────────────
#  Helper: single projection head
# ──────────────────────────────────────────────────────────────────────────────
def _make_proj_head(stem_dim: int, hidden_dim: int, projection_output: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(stem_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, projection_output),
        nn.BatchNorm1d(projection_output),
    )


# ──────────────────────────────────────────────────────────────────────────────
#  N-Expert MoE Encoder
# ──────────────────────────────────────────────────────────────────────────────
class MoENExpertEncoder(nn.Module):
    """
    Parameters
    ----------
    input_dim           : length of the raw 1-D signal window
    num_experts         : N — number of independent projection heads
    dropout_prob        : CNN stem dropout rate
    kernel_size / stride: stem CNN conv hyperparams
    output_dim          : hidden dim inside each projection head
    projection_output   : P — output dim of every expert (and h_out)
    router_temperature  : softmax temperature for the router gate
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int = 2,
        dropout_prob: float = 0.1,
        kernel_size: int = 7,
        stride: int = 1,
        output_dim: int = 64,
        projection_output: int = 32,
        router_temperature: float = 1.0,
    ):
        super().__init__()
        assert num_experts >= 1, "num_experts must be >= 1"

        self.num_experts       = num_experts
        self.output_dim        = output_dim         # D — stem hidden dim
        self.projection_output = projection_output  # P — expert output dim

        # ── Shared stem ──────────────────────────────────────────────────
        self.stem    = StemEncoder(input_dim, dropout_prob, kernel_size, stride)
        stem_dim     = self.stem.output_dim   # 256

        # ── Router ───────────────────────────────────────────────────────
        self.router  = Router(stem_dim, num_experts, router_temperature)

        # ── N expert projection heads ─────────────────────────────────────
        self.experts = nn.ModuleList([
            _make_proj_head(stem_dim, output_dim, projection_output)
            for _ in range(num_experts)
        ])

    # ------------------------------------------------------------------
    def forward(self, x):
        """
        x : [B, 1, L]  or  [B, L]

        Returns
        -------
        h_out  : [B, P]           router-weighted sum of all expert outputs
        z_list : list of N [B, P] individual expert outputs
        g      : [B, N]           routing weights (sum-to-1)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # 1. Shared stem  →  h [B, 256]
        h = self.stem(x)

        # 2. Router gate  →  g [B, N]
        g = self.router(h)

        # 3. Every expert runs independently on h
        z_list = [expert(h) for expert in self.experts]  # N × [B, P]

        # 4. Soft weighted combination
        #    h_out[b] = Σ_i  g[b, i] * z_list[i][b]
        z_stack = torch.stack(z_list, dim=1)              # [B, N, P]
        h_out   = (g.unsqueeze(-1) * z_stack).sum(dim=1)  # [B, P]

        return h_out, z_list, g