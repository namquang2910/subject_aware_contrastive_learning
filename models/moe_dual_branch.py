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
        super().__init__(device=device)
        self.device        = device
        self.lambda_inv    = lambda_inv
        self.lambda_spec   = lambda_spec
        self.lambda_shared = lambda_shared
        self.loss_fn       = None

        self.encoder = moe_encoder

    def get_parameters(self):
        return list(self.parameters()), []

    def forward(self, batch):
        assert self.loss_fn is not None, "Call set_loss_fn() before forward()."

        x1   = batch['x1']['x'].to(self.device, non_blocking=True).float()
        x2   = batch['x2']['x'].to(self.device, non_blocking=True).float()
        subj = batch['subject_id_int'].to(self.device, non_blocking=True).long()

        h_out1, z_inv1, z_spec1 = self.encoder(x1)
        h_out2, z_inv2, z_spec2 = self.encoder(x2)

        L_inv    = self.loss_fn(z_inv1, z_inv2)
        L_spec   = self.loss_fn(z_spec1, z_spec2, key_ids=subj)
        L_shared = self.loss_fn(h_out1, h_out2)

        total_loss = (
            self.lambda_inv    * L_inv    +
            self.lambda_spec   * L_spec   +
            self.lambda_shared * L_shared
        )

        return {
            "total_loss"     : total_loss,
            "L_sub_inv"      : L_inv,
            "L_sub_specific" : L_spec,
            "L_shared"       : L_shared,
        }


# ─────────────────────────────────────────────────────────────────
#  Fine-tuning / Linear Probe / Supervised / Random Init Model
# ─────────────────────────────────────────────────────────────────
class MoEFinetuneModel(Model):
    """
    Downstream model using the full MoE encoder.
    Modes: fine_tune, train_linear_projection
    """

    VALID_MODES = {"fine_tune", "train_linear_projection"}

    def __init__(
        self,
        moe_encoder,
        num_class:     int = 1,
        model_path:    str = None,
        training_mode: str = "fine_tune",
        device=None,
    ):
        super().__init__(device=device)

        self.device        = device
        self.loss_fn       = None
        self.encoder       = moe_encoder
        self.training_mode = training_mode

        assert training_mode in self.VALID_MODES, \
            f"Invalid mode '{training_mode}'. Must be one of {self.VALID_MODES}"

        # ── classifier head ────────────────────────────────────────────
        if training_mode == "train_linear_projection":
            self.classifier = nn.Linear(moe_encoder.output_dim, num_class)
        else:
            # fine_tune: full MoE combined output
            self.classifier = nn.Linear(moe_encoder.projection_output * 2, num_class)

        # ── weight loading ─────────────────────────────────────────────
        self._load_encoder(model_path)

        # ── freezing logic ─────────────────────────────────────────────
        if training_mode == "train_linear_projection":
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("[MoEFinetuneModel] train_linear_projection: "
                  "encoder frozen, training classifier head only")
        else:
            print("[MoEFinetuneModel] fine_tune: all layers trainable")

        self._sanity_check_frozen()

    # ── helpers ───────────────────────────────────────────────────────

    def _load_encoder(self, path: str):
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
            exit()
        print(f"[MoEFinetuneModel] Loaded encoder from '{path}'")

    def _sanity_check_frozen(self):
        if self.training_mode == "train_linear_projection":
            enc_frozen = all(not p.requires_grad for p in self.encoder.parameters())
            cls_trainable = all(p.requires_grad for p in self.classifier.parameters())
            assert enc_frozen, \
                "[MoEFinetuneModel] SANITY FAIL: encoder has trainable parameters!"
            assert cls_trainable, \
                "[MoEFinetuneModel] SANITY FAIL: classifier has frozen parameters!"
            print("[MoEFinetuneModel] Sanity check passed: "
                  "encoder frozen, classifier trainable")
        else:
            all_trainable = all(p.requires_grad for p in self.parameters())
            assert all_trainable, \
                "[MoEFinetuneModel] SANITY FAIL: fine_tune mode has frozen parameters!"
            print("[MoEFinetuneModel] Sanity check passed: all layers trainable")

    def get_parameters(self):
        enc_params = [p for p in self.encoder.parameters() if p.requires_grad]
        cls_params = list(self.classifier.parameters())
        return cls_params, enc_params

    # ── forward ───────────────────────────────────────────────────────

    def forward(self, data):
        assert self.loss_fn is not None, "Call set_loss_fn() before forward()."

        x = data['x'].to(self.device, non_blocking=True).float()
        y = data['y'].to(self.device, non_blocking=True).float()

        if x.dim() == 2:
            x = x.unsqueeze(1)

        if self.training_mode == "train_linear_projection":
            h, _, _, _, _ = self.encoder(x, return_embeddings=True)
        else:
            h, _, _ = self.encoder(x)

        y_hat = self.classifier(h)
        loss  = self.loss_fn(y_hat, self._prepare_targets(y))

        return {
            "total_loss": loss,
            "y_hat":      y_hat,
        }
