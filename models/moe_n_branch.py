import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from models.model import Model

# ──────────────────────────────────────────────────────────────────────────────
#  Pretraining model
# ──────────────────────────────────────────────────────────────────────────────
class MoENExpertPretrainModel(Model):
    """
    Parameters
    ----------
    moe_encoder   : MoENExpertEncoder instance
    num_subjects  : number of unique subjects (for adversarial head)
    grl_lambda    : GRL reversal strength on z_list[0]
    lambda_expert : weight for per-expert NCE losses
    lambda_shared : weight for NCE on the combined h_out
    lambda_orth   : weight for pairwise orthogonality penalty
    lambda_adv    : weight for adversarial subject loss on expert-0;
                    set to 0 to disable subject-adversarial training
    """

    def __init__(
        self,
        moe_encoder,
        grl_lambda:    float = 1.0,
        lambda_expert: float = 1.0,
        lambda_shared: float = 1.0,
        lambda_orth:   float = 0.1,
        device=None,
    ):
        super().__init__(device=device)

        self.grl_lambda    = grl_lambda
        self.lambda_expert = lambda_expert
        self.lambda_shared = lambda_shared
        self.lambda_orth   = lambda_orth
        self.loss_fn       = None   # NCELoss — injected via set_loss_fn()
        self.encoder     = moe_encoder
        N = moe_encoder.num_experts
        P = moe_encoder.projection_output

    # ------------------------------------------------------------------
    def get_parameters(self):
        return list(self.parameters()), []

    # ------------------------------------------------------------------
    def forward(self, batch):
        assert self.loss_fn is not None, "Call set_loss_fn() before forward()."

        x1   = batch['x1']['x'].to(self.device, non_blocking=True).float()
        x2   = batch['x2']['x'].to(self.device, non_blocking=True).float()
        subj = batch['subject_id_int'].to(self.device, non_blocking=True).long()

        # ── Encoder (stem → router → N experts → weighted sum) ──────────
        h_out1, z_list1, g1 = self.encoder(x1)
        h_out2, z_list2, g2 = self.encoder(x2)

        N = self.encoder.num_experts

        # ── Per-expert NCE losses ────────────────────────────────────────
        # Each expert sees the same-sample pair as positives (aug-invariant).
        # Downstream specialisation emerges from orthogonality + shared loss.
        L_experts = sum(
             self.loss_fn(z_list1[i], z_list2[i])
             for i in range(N)
         ) / N

        # ── Shared-output NCE (router supervision) ───────────────────────
        L_shared = self.loss_fn(h_out1, h_out2)


        total_loss = (
            self.lambda_expert * L_experts +
            self.lambda_shared * L_shared 
        )

        return {
            "total_loss" : total_loss,
             "L_experts"  : L_experts,
            "L_shared"   : L_shared,
        }


# ──────────────────────────────────────────────────────────────────────────────
#  Fine-tuning model
# ──────────────────────────────────────────────────────────────────────────────
class MoENExpertFinetuneModel(nn.Module):
    """
    Fine-tuning uses h_out — the router-weighted combination of all experts.

    h_out [B, P] = Σ_i  g_i · z_i   →  Linear(P → num_class)  →  BCE

    Freeze options
    --------------
    freeze_encoder : freeze the whole encoder (stem + router + all experts)
    freeze_stem    : freeze only the shared CNN stem
    freeze_experts : list of expert indices to freeze, e.g. [0, 1]
    freeze_router  : freeze only the router gate
    """

    def __init__(
        self,
        moe_encoder,
        num_class:      int  = 1,
        model_path:     str  = None,
        freeze_encoder: bool = False,
        freeze_stem:    bool = False,
        freeze_experts: list = None,   # e.g. [0, 2] to freeze experts 0 and 2
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

        # ── Freezing ──────────────────────────────────────────────────────
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        else:
            if freeze_stem:
                for p in self.encoder.stem.parameters():
                    p.requires_grad = False
            if freeze_router:
                for p in self.encoder.router.parameters():
                    p.requires_grad = False
            for idx in (freeze_experts or []):
                for p in self.encoder.experts[idx].parameters():
                    p.requires_grad = False

    # ------------------------------------------------------------------
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
            print(f"[MoENExpertFinetuneModel] Missing keys: {msg.missing_keys}")
        print(f"[MoENExpertFinetuneModel] Loaded encoder from '{path}'")

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

        # stem → router → N experts → weighted combination
        h_out, z_list, g = self.encoder(x)

        y_hat = self.classifier(h_out)
        loss  = self.loss_fn(y_hat, self._prepare_targets(y))

        # log per-expert mean gate weight for monitoring
        gate_info = {
            f"g_expert_{i}_mean": g[:, i].mean().detach()
            for i in range(self.encoder.num_experts)
        }

        return {
            "total_loss": loss,
            "y_hat"     : y_hat,
            **gate_info,
        }