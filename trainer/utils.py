import random
import os 
import json
from models import contrastive_model, finetune_builder, subject_invariant_model, subject_specific_model
import torch
import numpy as np
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from models.utils import get_base_encoder
import random
from datasets.wesad_dataset import WESADDataset
from datasets.psy_dataset import PsyDataset
from datasets.swell_dataset import SWELLDataset
from collections import defaultdict
from loss.cl_loss import NCELoss

class EarlyStopping:
    """Early-stopper on a scalar metric (lower is better)."""
    def __init__(self, min_delta=1e-3, patience=15, is_higher = True,enabled=True):
        self.enabled = bool(enabled)
        self.is_higher = bool(is_higher)
        self.min_delta = float(min_delta)
        self.patience = int(patience)
        self.best = None
        self.patience_counter = 0

    def step(self, value: float):
        """Update with latest value; return (should_stop: bool, improved: bool)."""
        improved = False
        if self.best is None:
            self.best = value
            return (False, True)

        if (self.best - value) > self.min_delta if not self.is_higher else (value - self.best) > self.min_delta:
            self.best = value
            self.patience_counter = 0
            improved = True
        else:
            self.patience_counter += 1

        if not self.enabled:
            return (False, improved)
        
        return (self.patience_counter >= self.patience, improved)

    def _best_loss_update(self, loss):
        if loss < self.best_loss:
            self.patience_counter = 0
            self.best_loss = loss
            return True
        return False


class LossMeter:
    def __init__(self):
        self.totals = defaultdict(float)
        self.count = 0

    def update(self, loss_dict):
        for k, v in loss_dict.items():
            if v is not None:
                self.totals[k] += v.detach()
        self.count += 1

    def average(self):
        return {k: v / max(1, self.count) for k, v in self.totals.items()}

class DummyLogger:
    """Logger that does nothing on non-zero ranks."""

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def compute_metrics(y_true, y_hat):
    # --- Ensure CPU numpy arrays ---
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.detach().cpu().numpy()

    # --- Compute metrics ---
    acc = accuracy_score(y_true, y_hat)
    precision = precision_score(y_true, y_hat, average='macro', zero_division=0)
    recall = recall_score(y_true, y_hat, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_hat, average='macro', zero_division=0)

    # --- Confusion matrix ---
    conf_mat = confusion_matrix(y_true, y_hat)

    return {
        'acc': round(acc, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'conf_mat': conf_mat
    }

def create_model(cfg, device):
    """
    Factory for all model types.

    New model_type values
    ─────────────────────
    "moe_dual_branch"  →  MoEPretrainModel   (pretraining)
    "moe_finetune"     →  MoEFinetuneModel   (fine-tuning)
    """
    from models.moe_dual_branch import (
        MoEDualBranchEncoder, MoEPretrainModel, MoEFinetuneModel
    )
    from models import contrastive_model, finetune_builder, subject_invariant_model, subject_specific_model
    from models.utils import get_base_encoder

    model_type = cfg["model_args"].get("model_type", "contrastive")

    # ── MoE dual-branch pretraining ───────────────────────────────────────
    if model_type == "moe_dual_branch":
        enc_args = cfg["model_args"]["encoder_args"]
        encoder  = MoEDualBranchEncoder(**enc_args)

        # num_subjects is injected by PreTrainer._build_dataloader()
        num_subjects = cfg["dataset_args"]["num_subjects"]

        model = MoEPretrainModel(
            encoder        = encoder,
            num_subjects   = num_subjects,
            projection_output = cfg["model_args"].get("projection_output", 32),
            grl_lambda     = cfg["model_args"].get("grl_lambda", 1.0),
            lambda_inv     = cfg["model_args"].get("lambda_inv",  1.0),
            lambda_spec    = cfg["model_args"].get("lambda_spec", 1.0),
            lambda_subj    = cfg["model_args"].get("lambda_subj", 1.0),
            lambda_adv     = cfg["model_args"].get("lambda_adv",  0.5),
            lambda_orth    = cfg["model_args"].get("lambda_orth", 0.1),
            device         = device,
        )
        return model

    # ── MoE fine-tuning ───────────────────────────────────────────────────
    if model_type == "moe_finetune":
        enc_args = cfg["model_args"]["encoder_args"]
        encoder  = MoEDualBranchEncoder(**enc_args)

        model = MoEFinetuneModel(
            encoder        = encoder,
            num_class      = cfg["model_args"].get("num_class", 1),
            model_path     = cfg["model_args"].get("model_path", None),
            freeze_encoder = cfg["model_args"].get("freeze_encoder", False),
            freeze_inv     = cfg["model_args"].get("freeze_inv", False),
            freeze_spec    = cfg["model_args"].get("freeze_spec", False),
            device         = device,
        )
        return model

    # ── existing model types (unchanged) ─────────────────────────────────
    enc_name = cfg["model_args"]["base_encoder"]
    enc_args = cfg["model_args"]["base_encoder_args"]
    encoder  = get_base_encoder(enc_name, enc_args)

    projection_output = cfg["model_args"].get("projection_output", 32)

    if model_type == "contrastive":
        return contrastive_model.ContrastiveModel(
            encoder, projection_output=projection_output, device=device
        )
    elif model_type == "subject_invariant":
        num_subjects = cfg["dataset_args"]["num_subjects"]
        grl_lambda   = cfg["model_args"].get("grl_lambda", 1.0)
        model = subject_invariant_model.SubjectInvariantContrastiveModel(
            encoder,
            projection_output=projection_output,
            num_subjects=num_subjects,
            grl_lambda=grl_lambda,
            device=device,
        )
        print(f"Created SubjectInvariantContrastiveModel with {num_subjects} subjects")
        return model
    elif model_type == "subject_specific":
        return subject_specific_model.SubjectSpecificContrastiveModel(
            encoder, projection_output=projection_output, device=device
        )
    elif model_type == "finetune":
        return finetune_builder.EncoderClassifierModel(
            base_encoder   = encoder,
            num_class      = 1,
            model_path     = cfg["model_args"]["model_path"],
            device         = device,
            freeze_encoder = cfg["model_args"].get("freeze_encoder", False),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def get_dataset(ds_args):
    data_name = ds_args.pop("data_name", None)
    if data_name =="WESADDataset":
        return WESADDataset(**ds_args)
    elif data_name == "PsychioNet":
        return PsyDataset(**ds_args)
    elif data_name == "SWELLDataset":
        return SWELLDataset(**ds_args)


def get_loss(name: str, loss_args: dict):
    if name == "NCE":
        return NCELoss(**loss_args)
    elif name == "BCE":
        return torch.nn.BCEWithLogitsLoss()
    raise ValueError(f"Unknown loss: {name}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logger(output_dir):
    """Logs to both console and file: output_dir/train.log"""
    log_path = os.path.join(output_dir, "train.log")
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    # File
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Make sure 3rd-party libs don’t spam DEBUG
    logging.getLogger().setLevel(logging.WARNING)
    return logger

def save_config_file(config_dict, output_dir):
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)
