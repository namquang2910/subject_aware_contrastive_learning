
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from models.model import Model


# ─────────────────────────────────────────────────────────────────
#  Pretraining Model
# ─────────────────────────────────────────────────────────────────
class MoEPretrainModel(Model):

    def __init__(
        self,
        moe_encoder,
        lambda_inv:    float = 1.0,
        lambda_spec:   float = 1.0,
        lambda_shared: float = 1.0,
        device=None,
    ):
        super().__init__(device= device)
        self.device        = device
        self.grl_lambda    = grl_lambda
        self.lambda_inv    = lambda_inv
        self.lambda_spec   = lambda_spec
        self.lambda_shared = lambda_shared
        self.loss_fn       = None   # NCELoss — set via set_loss_fn()

        self.encoder = moe_encoder
        P = moe_encoder.projection_output   # expert output dim


    def get_parameters(self):
        return list(self.parameters()), []

    def forward(self, batch):
        assert self.loss_fn is not None, "Call set_loss_fn() before forward()."

        x1   = batch['x1']['x'].to(self.device, non_blocking=True).float()
        x2   = batch['x2']['x'].to(self.device, non_blocking=True).float()
        subj = batch['subject_id_int'].to(self.device, non_blocking=True).long()

        # MoeDueEnocder
        ## stem -> router -> 2 expert heads -> Weight combine
        h_out1, z_inv1, z_spec1 = self.encoder(x1)
        h_out2, z_inv2, z_spec2  = self.encoder(x2)

        # Subject Invariant Expert
        L_inv = self.loss_fn(z_inv1, z_inv2)

        # Subject Specific Expert
        ## Treating the same subject as positive
        L_spec = self.loss_fn(z_spec1, z_spec2, key_ids=subj)

        # Weight Combination using the router
        L_shared = self.loss_fn(h_out1, h_out2)


        total_loss = (
            self.lambda_inv    * L_inv    +
            self.lambda_spec   * L_spec   +
            self.lambda_shared * L_shared
        )

        return {
            "total_loss"  : total_loss,
            "L_sub_inv"       : L_inv,
            "L_sub_specific"      : L_spec,
            "L_shared"    : L_shared,
        }


# ─────────────────────────────────────────────────────────────────
#  Fine-tuning Model
# ─────────────────────────────────────────────────────────────────
class MoEFinetuneModel(nn.Module):
    """
    Fine-tuning uses h_out — the router-weighted combination.

    h_out [B, P] = g_inv·z_inv + g_spec·z_spec
    → Linear(P → num_class) → BCE

    Same classifier size as any single-encoder baseline (fair comparison).

    freeze options:
        freeze_encoder — freeze stem + router + both experts
        freeze_stem    — freeze only the shared CNN stem
        freeze_inv     — freeze only the invariant projection head
        freeze_spec    — freeze only the subject-specific projection head
        freeze_router  — freeze only the router gate
    """

    def __init__(
        self,
        moe_encoder,
        num_class:      int  = 1,
        model_path:     str  = None,
        freeze_encoder: bool = False,
        freeze_stem:    bool = False,
        freeze_inv:     bool = False,
        freeze_spec:    bool = False,
        freeze_router:  bool = False,
        device=None,
    ):
        super().__init__()
        self.device  = device
        self.loss_fn = None
        self.encoder = moe_encoder

        P = moe_encoder.projection_output
        self.classifier = nn.Linear(P, num_class)

        if model_path:
            self._load_encoder(model_path)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        else:
            if freeze_stem:
                for p in self.encoder.stem.parameters():
                    p.requires_grad = False
            if freeze_inv:
                for p in self.encoder.proj_inv.parameters():
                    p.requires_grad = False
            if freeze_spec:
                for p in self.encoder.proj_spec.parameters():
                    p.requires_grad = False
            if freeze_router:
                for p in self.encoder.router.parameters():
                    p.requires_grad = False

    def _load_encoder(self, path: str):
        import os
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No checkpoint at '{path}'")
        ckpt   = torch.load(path, map_location="cpu", weights_only=False)
        sd     = ckpt.get("state_dict", ckpt)
        enc_sd = {
            k.replace("module.", "", 1)[len("encoder."):]: v
            for k, v in sd.items()
            if k.replace("module.", "", 1).startswith("encoder.")
        }
        msg = self.encoder.load_state_dict(enc_sd, strict=False)
        if msg.missing_keys:
            print(f"[MoEFinetuneModel] Missing keys: {msg.missing_keys}")
        print(f"[MoEFinetuneModel] Loaded encoder from '{path}'")

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
            x = x.unsqueeze(1)

        # stem → router → expert heads → combine
        h_out, _, _ = self.encoder(x)

        y_hat = self.classifier(h_out)
        loss  = self.loss_fn(y_hat, self._prepare_targets(y))

        return {
            "total_loss"  : loss,
            "y_hat"       : y_hat
        }