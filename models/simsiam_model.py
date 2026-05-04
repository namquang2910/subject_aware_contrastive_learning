import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model import Model

# ══════════════════════════════════════════════════════════════════════════════
#  SimSiam
# ══════════════════════════════════════════════════════════════════════════════

def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """Two-layer MLP (optionally with BN after first linear)."""
    return nn.Sequential(nn.Linear(in_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_dim, out_dim))


def _simsiam_projector(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """SimSiam 3-layer projector with BN (no ReLU on last layer, BN on output)."""
    return nn.Sequential(
        nn.Linear(in_dim,    hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim,   bias=False),
        nn.BatchNorm1d(out_dim, affine=False),   # no learnable params on last BN
    )

class SimSiamPretrainModel(Model):
    """
    Exploring Simple Siamese Representation Learning (Chen & He, 2021).
    """

    def __init__(
        self,
        base_encoder,
        projection_output: int = 256,
        hidden_dim:        int = 256,
        device=None,
    ):
        super().__init__(device=device)
        stem_dim = base_encoder.last_dim

        self.encoder   = copy.deepcopy(base_encoder)
        self.projector = _simsiam_projector(stem_dim, hidden_dim, projection_output)
        # Predictor: 2-layer MLP (hidden_dim // 4 following original paper ratio)
        pred_hidden = max(64, hidden_dim // 4)
        self.predictor = _mlp(projection_output, pred_hidden, projection_output)

        self.projection_output = projection_output

    # ------------------------------------------------------------------
    def get_parameters(self):
        params = (
            list(self.encoder.parameters())
            + list(self.projector.parameters())
            + list(self.predictor.parameters())
        )
        return params, []

    # ------------------------------------------------------------------
    def forward(self, batch: dict) -> dict:
        x1 = batch['x1']['x'].to(self.device, non_blocking=True).float()
        x2 = batch['x2']['x'].to(self.device, non_blocking=True).float()

        if x1.dim() == 2:
            x1 = x1.unsqueeze(1)
            x2 = x2.unsqueeze(1)

        # Encoder + projector
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))

        # Predictor (applied to both views)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # Stop-gradient on z (detach target side)
        loss = (self.loss_fn(p1, z2.detach()) + self.loss_fn(p2, z1.detach())) * 0.5

        return {"total_loss": loss}


# ──────────────────────────────────────────────────────────────────────────────
#  SimSiam fine-tuning
# ──────────────────────────────────────────────────────────────────────────────

class SimSiamFinetuneModel(Model):
    """
    Fine-tuning with the SimSiam encoder.

    Checkpoint keys start with 'encoder.' (produced by SimSiamPretrainModel).
    """

    def __init__(
        self,
        base_encoder,
        num_class:      int  = 1,
        model_path:     str  = None,
        freeze_encoder: bool = False,
        mode = "fine_tune",  # one of "fine_tune", "train_linear", "supervised", "random_init"
        device=None,
    ):
        super().__init__(device=device)
        self.device  = device
        self.loss_fn = None

        self.encoder    = base_encoder
        self.classifier = nn.Linear(self.encoder.output_dim, num_class)

        if model_path:
            self._load_encoder(model_path)
            
        if mode == "train_linear":
            print("[SimSiamFinetuneModel] train_linear: "
                  "pretrained encoder frozen, training linear head only")
            freeze_encoder = True

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.check_frozen(self.encoder)
    # ------------------------------------------------------------------
    def _load_encoder(self, path: str):
        import os
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No checkpoint at '{path}'")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd   = ckpt.get("state_dict", ckpt)

        enc_sd = {
            k[len("encoder."):]: v
            for k, v in sd.items()
            if k.startswith("encoder.")
        }
        msg = self.encoder.load_state_dict(enc_sd, strict=False)
        if msg.missing_keys:
            print(f"[SimSiamFinetuneModel] Missing keys: {msg.missing_keys}")
        print(f"[SimSiamFinetuneModel] Loaded encoder from '{path}'")

    # ------------------------------------------------------------------
    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    def get_parameters(self):
        return list(self.classifier.parameters()), list(self.encoder.parameters())

    def to(self, device):
        self.device = device
        return super().to(device)

    def _prepare_targets(self, y):
        if y.dtype == torch.double:
            y = y.float()
        if y.dim() == 1:
            y = y[:, None].float()
        return y.to(self.device)

    def forward(self, data: dict) -> dict:
        assert self.loss_fn is not None, "Call set_loss_fn() before forward()."
        x = data['x'].to(self.device, non_blocking=True).float()
        y = data['y'].to(self.device, non_blocking=True).float()

        if x.dim() == 2:
            x = x.unsqueeze(1)

        h, z     = self.encoder(x, return_embedding=True)
        y_hat = self.classifier(z)
        loss  = self.loss_fn(y_hat, self._prepare_targets(y))

        return {"total_loss": loss, "y_hat": y_hat}