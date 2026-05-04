import copy
import torch
import torch.nn as nn
import math
from models.model import Model


# ──────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """Two-layer MLP (optionally with BN after first linear)."""
    return nn.Sequential(nn.Linear(in_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_dim, out_dim))



# ──────────────────────────────────────────────────────────────────────────────
#  EMA helper (used by BYOL)
# ──────────────────────────────────────────────────────────────────────────────

class EMA:
    """Exponential Moving Average for target-network weights."""
    def __init__(self, base_beta: float = 0.99, max_steps: int = None):
        self.base_beta = base_beta
        self.beta = base_beta
        self.max_steps = max_steps
        self.current_step = 0

    def update_beta(self):
        if self.max_steps is not None:
            self.beta = 1 - (1 - self.base_beta) * ( math.cos(math.pi * self.current_step / self.max_steps) + 1) / 2
    
    @torch.no_grad()
    def update(self, online: nn.Module, target: nn.Module):
        for o_p, t_p in zip(online.parameters(), target.parameters()):
            t_p.data = self.beta * t_p.data + (1.0 - self.beta) * o_p.data
            


# ══════════════════════════════════════════════════════════════════════════════
#  BYOL
# ══════════════════════════════════════════════════════════════════════════════

class BYOLPretrainModel(Model):
    def __init__(
        self,
        base_encoder,
        projection_output: int   = 256,
        hidden_dim:        int   = 512,
        ema_decay:         float = 0.996,
        total_steps:       int   = None,
        device=None,
    ):
        super().__init__(device=device)
        stem_dim = base_encoder.output_dim

        # ── Online network ────────────────────────────────────────────────
        self.online_encoder   = copy.deepcopy(base_encoder)
        self.online_projector = _mlp(stem_dim, hidden_dim, projection_output)
        self.online_predictor = _mlp(projection_output, hidden_dim // 2, projection_output)

        # ── Target network (EMA copy, no grad) ───────────────────────────
        self.target_encoder   = copy.deepcopy(base_encoder)
        self.target_projector = _mlp(stem_dim, hidden_dim, projection_output)
        for p in (*self.target_encoder.parameters(),
                  *self.target_projector.parameters()):
            p.requires_grad = False

        self.ema = EMA(base_beta=ema_decay, max_steps = total_steps)
        self.projection_output = projection_output

    # ------------------------------------------------------------------
    def _online_forward(self, x: torch.Tensor):
        h = self.online_encoder(x)
        z = self.online_projector(h)
        p = self.online_predictor(z)
        return p

    @torch.no_grad()
    def _target_forward(self, x: torch.Tensor):
        h = self.target_encoder(x)
        z = self.target_projector(h)
        return z.detach()

    # ------------------------------------------------------------------
    def update_moving_average(self):
        """Call once per step, after optimizer.step()."""
        #Update the new beta
        self.ema.update_beta() 
        #Update the encoder and target
        self.ema.update(self.online_encoder,   self.target_encoder)
        self.ema.update(self.online_projector, self.target_projector)
        #Update the current steps
        self.ema.current_step += 1
        return self.ema.beta
    # ------------------------------------------------------------------
    def get_parameters(self):
        params = (
            list(self.online_encoder.parameters())
            + list(self.online_projector.parameters())
            + list(self.online_predictor.parameters())
        )
        return params, []

    # ------------------------------------------------------------------
    def forward(self, batch: dict) -> dict:
        x1 = batch['x1']['x'].to(self.device, non_blocking=True).float()
        x2 = batch['x2']['x'].to(self.device, non_blocking=True).float()

        if x1.dim() == 2:
            x1 = x1.unsqueeze(1)
            x2 = x2.unsqueeze(1)

        p1 = self._online_forward(x1)
        p2 = self._online_forward(x2)

        with torch.no_grad():
            z1 = self._target_forward(x1)
            z2 = self._target_forward(x2)

        loss = (self.loss_fn(p1, z2) + self.loss_fn(p2, z1)) * 0.5

        return {"total_loss": loss}


# ──────────────────────────────────────────────────────────────────────────────
#  BYOL fine-tuning  (encoder only, same classifier head as other methods)
# ──────────────────────────────────────────────────────────────────────────────

class BYOLFinetuneModel(Model):
    """
    Loads the BYOL online encoder weights and attaches a linear classifier.
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
            print("[BYOLFinetuneModel] train_linear: "
                  "pretrained encoder frozen, training linear head only")
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

        # Strip 'online_encoder.' prefix produced by BYOLPretrainModel
        enc_sd = {
            k[len("online_encoder."):]: v
            for k, v in sd.items()
            if k.startswith("online_encoder.")
        }
        if not enc_sd:
            # Fallback: try generic 'encoder.' prefix
            enc_sd = {
                k[len("encoder."):]: v
                for k, v in sd.items()
                if k.startswith("encoder.")
            }
        msg = self.encoder.load_state_dict(enc_sd, strict=False)
        if msg.missing_keys:
            print(f"[BYOLFinetuneModel] Missing keys: {msg.missing_keys}")
        print(f"[BYOLFinetuneModel] Loaded encoder from '{path}'")

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

