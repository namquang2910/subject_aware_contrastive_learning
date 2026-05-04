from models.net.CNNEncoder import CNNEncoder
from models.model import Model
import torch
import torch.nn as nn
import os

class StemFinetuneModel(Model):
    """
    Downstream model using only the CNN stem encoder.
    Modes: supervised, random_init, train_linear

    Architecture:
        cnn_layers  (frozen for train_linear/random_init)
            ↓
        linear_layer with ReLU  (frozen for train_linear/random_init)
            ↓
        classifier  nn.Linear(last_dim, num_class)  (always trainable)
    """

    VALID_MODES = {"supervised", "random_init", "train_linear"}

    def __init__(
        self,
        base_encoder: CNNEncoder,
        num_class:     int = 1,
        model_path:    str = None,
        training_mode: str = "supervised",
        device=None,
    ):
        super().__init__(device=device)

        self.device        = device
        self.loss_fn       = None
        self.encoder       = base_encoder
        self.training_mode = training_mode

        assert training_mode in self.VALID_MODES, \
            f"Invalid mode '{training_mode}'. Must be one of {self.VALID_MODES}"

        # ── classifier head ────────────────────────────────────────────
        # sits on top of linear_layer (last_dim) since linear_layer has ReLU
        self.classifier = nn.Linear(base_encoder.cnn_output_dim, num_class)

        # ── weight loading ─────────────────────────────────────────────
        if training_mode == "train_linear":
            self._load_encoder(model_path)

        # ── freezing logic ─────────────────────────────────────────────
        if training_mode in {"train_linear", "random_init"}:
            # freeze both cnn_layers and linear_layer
            for p in self.encoder.cnn_layers.parameters():
                p.requires_grad = False
            # for p in self.encoder.linear_layer.parameters():
            #     p.requires_grad = False
            print(f"[StemFinetuneModel] {training_mode}: "
                  "cnn_layers + linear_layer frozen, classifier trainable")

        elif training_mode == "supervised":
            print("[StemFinetuneModel] supervised: "
                  "all layers trainable, training from scratch")

        self._sanity_check_frozen()

    # ── helpers ───────────────────────────────────────────────────────

    def _load_encoder(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No checkpoint at '{path}'")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd   = ckpt.get("state_dict", ckpt)

        stem_sd = {
            k.replace("module.", "", 1)[len("encoder.stem."):]: v
            for k, v in sd.items()
            if k.replace("module.", "", 1).startswith("encoder.stem.")
        }

        msg = self.encoder.load_state_dict(stem_sd, strict=False)
        if msg.missing_keys:
            print(f"[StemFinetuneModel] Missing keys: {msg.missing_keys}")
            exit()
        print(f"[StemFinetuneModel] Loaded CNNEncoder from '{path}'")

    def _sanity_check_frozen(self):
        if self.training_mode in {"train_linear", "random_init"}:
            cnn_frozen    = all(
                not p.requires_grad
                for p in self.encoder.cnn_layers.parameters()
            )
            # linear_frozen = all(
            #     not p.requires_grad
            #     for p in self.encoder.linear_layer.parameters()
            # )
            cls_trainable = all(
                p.requires_grad
                for p in self.classifier.parameters()
            )
            assert cnn_frozen, \
                "[StemFinetuneModel] SANITY FAIL: cnn_layers has trainable parameters!"
            # assert linear_frozen, \
            #     "[StemFinetuneModel] SANITY FAIL: linear_layer has trainable parameters!"
            assert cls_trainable, \
                "[StemFinetuneModel] SANITY FAIL: classifier has frozen parameters!"
            print("[StemFinetuneModel] Sanity check passed: "
                  "cnn_layers + linear_layer frozen, classifier trainable")

        elif self.training_mode == "supervised":
            all_trainable = all(p.requires_grad for p in self.encoder.parameters())
            assert all_trainable, \
                "[StemFinetuneModel] SANITY FAIL: supervised mode has frozen parameters!"
            print("[StemFinetuneModel] Sanity check passed: all layers trainable")

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

        h, z     = self.encoder(x, return_embedding=True)        # (B, last_dim=256) — after ReLU
        y_hat = self.classifier(z)     # (B, num_class)    — raw logits
        loss  = self.loss_fn(y_hat, self._prepare_targets(y))

        return {
            "total_loss": loss,
            "y_hat":      y_hat,
        }