import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from models.net.CNNEncoder import CNNEncoder


# ─────────────────────────────────────────────
#  Gradient Reversal Layer
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
#  Lightweight Expert Branch (reuses CNN stem
#  output and adds its own deeper blocks)
# ─────────────────────────────────────────────
class ExpertBranch(nn.Module):
    """
    Takes stem feature map [B, C_stem, L'] and produces
    an embedding [B, output_dim].

    Architecture:
        Conv1d(C_stem → 128) → BN → ReLU → MaxPool
        Conv1d(128    → 256) → BN → ReLU → MaxPool
        AdaptiveAvgPool → flatten
        Linear(256 → output_dim) → ReLU
    """
    def __init__(self, stem_channels: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.Conv1d(stem_channels, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: [B, C_stem, L']
        h = self.layers(x)          # [B, 256, L'']
        h = self.gap(h).squeeze(-1) # [B, 256]
        return self.fc(h)           # [B, output_dim]


# ─────────────────────────────────────────────
#  Router Network
# ─────────────────────────────────────────────
class Router(nn.Module):
    """
    Produces soft gating weights g = [g_inv, g_spec] ∈ (0,1)² summing to 1.

    Input  : flattened stem features [B, flat_dim]
    Output : [B, 2]  (column 0 = weight for invariant branch,
                      column 1 = weight for subject-specific branch)

    temperature τ: lower → harder routing (towards hard MoE),
                   higher → softer, more equal mixing.
    """
    def __init__(self, flat_dim: int, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.gate = nn.Sequential(
            nn.Linear(flat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, x_flat):
        # x_flat: [B, flat_dim]
        logits = self.gate(x_flat)              # [B, 2]
        return F.softmax(logits / self.temperature, dim=-1)  # [B, 2]


# ─────────────────────────────────────────────
#  MoE Dual-Branch Encoder  (public API)
# ─────────────────────────────────────────────
class MoEDualBranchEncoder(nn.Module):
    """
    Shared stem → Router → [InvariantExpert, SubjectSpecificExpert]
    → weighted-sum output h  [B, output_dim]

    Exposes:
        forward(x) → h, h_inv, h_spec, g
        output_dim  (for downstream heads)
    """
    def __init__(
        self,
        input_dim: int   = 1280,
        stem_channels: int = 64,    # output channels of the shared stem
        output_dim: int  = 64,
        dropout: float   = 0.1,
        router_temperature: float = 1.0,
    ):
        super().__init__()
        self.output_dim = output_dim

        # ── Shared stem (2 CNN blocks) ────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=10, stride=1, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(32, stem_channels, kernel_size=10, stride=1, padding=4),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

        # figure out stem output length for the router
        self._input_dim   = input_dim
        self._stem_ch     = stem_channels
        # compute dynamically in forward so we don't need to hard-code
        self._stem_flat   = None   # filled on first forward

        # ── Expert branches ──────────────────────────────────────
        self.expert_inv  = ExpertBranch(stem_channels, output_dim, dropout)
        self.expert_spec = ExpertBranch(stem_channels, output_dim, dropout)

        # ── Router ───────────────────────────────────────────────
        # flat_dim computed lazily on first call
        self._router: Router = None
        self._router_temperature = router_temperature

    # ── lazy router init ─────────────────────────────────────────
    def _init_router(self, flat_dim: int):
        self._router = Router(flat_dim, self._router_temperature).to(
            next(self.stem.parameters()).device
        )

    def forward(self, x):
        """
        x : [B, 1, L]  or  [B, L]

        Returns
        -------
        h      : [B, output_dim]   weighted mixture embedding
        h_inv  : [B, output_dim]   invariant branch output
        h_spec : [B, output_dim]   subject-specific branch output
        g      : [B, 2]            router weights [g_inv, g_spec]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)   # [B, 1, L]

        # ── shared stem ──────────────────────────────────────────
        s = self.stem(x)         # [B, stem_channels, L']
        B = s.size(0)
        s_flat = s.view(B, -1)   # [B, stem_channels * L']

        # ── lazy router creation ──────────────────────────────────
        if self._router is None:
            self._init_router(s_flat.size(1))

        # ── routing weights ───────────────────────────────────────
        g = self._router(s_flat)            # [B, 2]
        g_inv  = g[:, 0:1]                  # [B, 1]
        g_spec = g[:, 1:2]                  # [B, 1]

        # ── expert branches ──────────────────────────────────────
        h_inv  = self.expert_inv(s)         # [B, D]
        h_spec = self.expert_spec(s)        # [B, D]

        # ── weighted mixture ─────────────────────────────────────
        h = g_inv * h_inv + g_spec * h_spec # [B, D]

        return h, h_inv, h_spec, g


# ─────────────────────────────────────────────────────────────────────────────
#  Pre-training Model
# ─────────────────────────────────────────────────────────────────────────────
class MoEPretrainModel(nn.Module):
    """
    Pretraining wrapper around MoEDualBranchEncoder.

    Losses (each optionally weighted):
    ┌────────────────────────────────────────────────────────────────────┐
    │  L_total = λ_inv  · NCE(proj_inv(h_inv_v1),  proj_inv(h_inv_v2))  │
    │          + λ_spec · NCE(proj_spec(h_spec_v1), proj_spec(h_spec_v2)│
    │                         with same-subject positives)               │
    │          + λ_subj · CE(subj_head(h_spec), subject_id)             │
    │          + λ_adv  · CE(subj_adv(GRL(h_inv)), subject_id)          │
    │          + λ_orth · ||h_inv^T h_spec||_F / B                      │
    └────────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        encoder: MoEDualBranchEncoder,
        num_subjects: int,
        projection_output: int = 32,
        grl_lambda: float       = 1.0,
        lambda_inv: float       = 1.0,
        lambda_spec: float      = 1.0,
        lambda_subj: float      = 1.0,
        lambda_adv: float       = 0.5,
        lambda_orth: float      = 0.1,
        device=None,
    ):
        super().__init__()
        self.device         = device
        self.grl_lambda     = grl_lambda
        self.lambda_inv     = lambda_inv
        self.lambda_spec    = lambda_spec
        self.lambda_subj    = lambda_subj
        self.lambda_adv     = lambda_adv
        self.lambda_orth    = lambda_orth
        self.loss_fn        = None   # set externally via set_loss_fn()

        D = encoder.output_dim
        self.encoder = encoder

        # Projection heads
        self.proj_inv = nn.Sequential(
            nn.Linear(D, projection_output),
            nn.BatchNorm1d(projection_output),
        )
        self.proj_spec = nn.Sequential(
            nn.Linear(D, projection_output),
            nn.BatchNorm1d(projection_output),
        )

        # Subject classification head (on h_spec)
        self.subj_criterion = nn.CrossEntropyLoss()

        # Adversarial subject head (on GRL(h_inv)) – same architecture
        self.subj_adv = nn.Sequential(
            nn.Linear(D, D * 2),
            nn.ReLU(inplace=True),
            nn.Linear(D * 2, num_subjects),
        )

    # ── called by Trainer ────────────────────────────────────────
    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    def get_parameters(self):
        return list(self.parameters()), []

    def to(self, device):
        self.device = device
        return super().to(device)

    # ── orthogonality loss ───────────────────────────────────────
    @staticmethod
    def _orthogonality_loss(h_inv, h_spec):
        """
        Penalise alignment between the two branches.
        ||h_inv^T h_spec||_F / B
        """
        h_inv_n  = F.normalize(h_inv,  dim=1)
        h_spec_n = F.normalize(h_spec, dim=1)
        gram = torch.mm(h_inv_n.T, h_spec_n)   # [D, D]
        return torch.norm(gram, p='fro') / h_inv.size(0)

    def forward(self, batch):
        assert self.loss_fn is not None, "Call set_loss_fn() before forward()."

        # ── unpack batch ─────────────────────────────────────────
        x1 = batch['x1']['x'].to(self.device, non_blocking=True).float()
        x2 = batch['x2']['x'].to(self.device, non_blocking=True).float()
        subj = batch['subject_id_int'].to(self.device, non_blocking=True).long()

        # ── encode both views ────────────────────────────────────
        _, h_inv1, h_spec1, g1 = self.encoder(x1)
        _, h_inv2, h_spec2, g2 = self.encoder(x2)

        # ── (1) Invariant contrastive loss  (cross-subject) ───────
        z_inv1 = self.proj_inv(h_inv1)
        z_inv2 = self.proj_inv(h_inv2)
        # no key_ids → each sample is its own positive (classic NCE)
        L_inv = self.loss_fn(z_inv1, z_inv2)

        # ── (2) Subject-specific contrastive loss  ────────────────
        z_spec1 = self.proj_spec(h_spec1)
        z_spec2 = self.proj_spec(h_spec2)
        # pass subject ids so same-subject pairs = positives
        L_spec = self.loss_fn(z_spec1, z_spec2, key_ids=subj)


        # ── (4) Adversarial loss on h_inv (via GRL) ───────────────
        h_inv1_rev  = grad_reverse(h_inv1, self.grl_lambda)
        adv_logits  = self.subj_adv(h_inv1_rev)
        L_adv       = self.subj_criterion(adv_logits, subj)

        # ── (5) Orthogonality regulariser ─────────────────────────
        L_orth = self._orthogonality_loss(h_inv1, h_spec1)

        # ── total loss ────────────────────────────────────────────
        total = (
            self.lambda_inv  * L_inv  +
            self.lambda_spec * L_spec +
            self.lambda_adv  * L_adv  +
            self.lambda_orth * L_orth
        )

        return {
            "total_loss"  : total,
            "L_inv"       : L_inv,
            "L_spec"      : L_spec,
            "L_adv"       : L_adv,
            "L_orth"      : L_orth,
            "g_inv_mean"  : g1[:, 0].mean().detach(),
            "g_spec_mean" : g1[:, 1].mean().detach(),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Fine-tuning Model
# ─────────────────────────────────────────────────────────────────────────────
class MoEFinetuneModel(nn.Module):
    """
    Fine-tuning wrapper.

    The classifier sees the *concatenation* of both branch embeddings:
        h_cat = [h_inv ; h_spec]   [B, 2·output_dim]
        → Linear(2D → num_class)

    Loading:  pass the path to an encoder_best_*.pt checkpoint;
              only the encoder weights are loaded (proj heads discarded).

    freeze_encoder : if True, encoder weights are frozen (linear probe mode)
    freeze_inv     : freeze only the invariant branch
    freeze_spec    : freeze only the subject-specific branch
    """

    def __init__(
        self,
        encoder: MoEDualBranchEncoder,
        num_class: int  = 1,
        model_path: str = None,
        freeze_encoder: bool = False,
        freeze_inv: bool     = False,
        freeze_spec: bool    = False,
        device=None,
    ):
        super().__init__()
        self.device    = device
        self.loss_fn   = None
        self.encoder   = encoder

        D = encoder.output_dim
        self.classifier = nn.Linear(2 * D, num_class)

        # ── load pre-trained encoder ─────────────────────────────
        if model_path:
            self._load_encoder(model_path)

        # ── selective freezing ───────────────────────────────────
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        else:
            if freeze_inv:
                for p in self.encoder.expert_inv.parameters():
                    p.requires_grad = False
            if freeze_spec:
                for p in self.encoder.expert_spec.parameters():
                    p.requires_grad = False

    # ── weight loading ───────────────────────────────────────────
    def _load_encoder(self, path: str):
        import os
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No checkpoint at '{path}'")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd   = ckpt.get("state_dict", ckpt)

        enc_sd = {}
        for k, v in sd.items():
            k = k.replace("module.", "", 1)
            if k.startswith("encoder."):
                enc_sd[k[len("encoder."):]] = v
        msg = self.encoder.load_state_dict(enc_sd, strict=False)
        if msg.missing_keys:
            # only raise if non-router keys are missing
            non_router = [k for k in msg.missing_keys if "_router" not in k]
            if non_router:
                raise ValueError(f"Missing encoder keys: {non_router}")
        print(f"[MoEFinetuneModel] Loaded encoder from '{path}'")

    # ── called by Trainer ────────────────────────────────────────
    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    def get_parameters(self):
        enc_params = list(self.encoder.parameters())
        cls_params = list(self.classifier.parameters())
        return cls_params, enc_params

    def to(self, device):
        self.device = device
        return super().to(device)

    def _prepare_targets(self, y):
        if y.dtype == torch.double:
            y = y.float()
        if y.dim() == 1:
            y = y[:, None].float()
        return y.to(self.device)

    def forward(self, data):
        assert self.loss_fn is not None, "Call set_loss_fn() before forward()."

        x = data['x'].to(self.device, non_blocking=True).float()
        y = data['y'].to(self.device, non_blocking=True).float()

        if x.dim() == 2:
            x = x.unsqueeze(1)   # [B, 1, L]

        # ── encode ───────────────────────────────────────────────
        _, h_inv, h_spec, g = self.encoder(x)

        # ── concatenate both branches ────────────────────────────
        h_cat = torch.cat([h_inv, h_spec], dim=1)   # [B, 2D]

        y_hat = self.classifier(h_cat)               # [B, num_class]
        y     = self._prepare_targets(y)
        loss  = self.loss_fn(y_hat, y)

        return {
            "total_loss"  : loss,
            "y_hat"       : y_hat,
            "g_inv_mean"  : g[:, 0].mean().detach(),
            "g_spec_mean" : g[:, 1].mean().detach(),
        }