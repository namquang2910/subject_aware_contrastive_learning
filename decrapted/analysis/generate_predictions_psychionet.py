"""
generate_predictions_psychionet.py

Loads the three pretrained checkpoints (BYOL, SimSiam, DualBranch-no-scale)
that were pretrained on PsychioNet, fine-tunes a frozen linear head on each
WESAD LOSO fold, and saves per-sample (y_pred, y_true, fold_id) to disk for
subsequent pairwise bootstrap analysis.

Usage:
    conda run -n pytorch310 python generate_predictions_psychionet.py

Outputs:
    save/predictions/psychionet_byol_preds.pkl
    save/predictions/psychionet_simsiam_preds.pkl
    save/predictions/psychionet_dual_branch_no_scale_preds.pkl
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

REPO = "/home/s223149341/SSL-invariance-Subject_Project_model/subject_aware_contrastive_learning"
sys.path.insert(0, REPO)

from models.net.CNNEncoder import CNNEncoder
from models.net.moe_encoder import MoEDualBranchEncoder
from datasets.wesad_dataset import WESADDataset

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = os.path.join(REPO, "save/predictions")
os.makedirs(SAVE_DIR, exist_ok=True)

WESAD_DATA = "/home/s223149341/SSL-invariance-Subject_Project_model/data/WESAD/wesad_10_05_no_standardize"
LOSO_DIR   = os.path.join(REPO, "datasets/process_dataset/WESAD_LOSO")

CHECKPOINTS = {
    "byol": os.path.join(REPO, "save/BYOL/byol_PsychioNet_0/pretrain/encoder_best_.pt"),
    "simsiam": os.path.join(REPO, "save/Simsiam/simsiam_PsychioNet_0/pretrain/encoder_best_.pt"),
    "dual_branch_no_scale": os.path.join(REPO, "save/dual_branch_no_scale/moe_dual_branch_PsychioNet_0/pretrain/encoder_best_.pt"),
}

CNN_ARGS = dict(input_dim=1280, kernel_size=10, dropout_prob=0.0, stride=1)
MOE_ARGS = dict(input_dim=1280, kernel_size=10, output_dim=64, stride=1,
                dropout_prob=0.0, projection_output=32)

# Linear probe training settings
FINETUNE_EPOCHS    = 100
FINETUNE_LR        = 1e-3
FINETUNE_BATCH     = 64
FINETUNE_WORKERS   = 4
SUBSAMPLE_FRAC     = 0.01   # match original config (1% of train data)


# ── Encoder wrappers with GAP head ───────────────────────────────────────────

class CNNFeatureExtractor(nn.Module):
    """CNNEncoder → global-average-pool → 256-dim feature."""
    def __init__(self, cnn: CNNEncoder):
        super().__init__()
        self.cnn = cnn
        self.feature_dim = cnn.last_dim   # 256

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.cnn.cnn_layers(x)        # (B, 256, T)
        return h.mean(dim=-1)             # (B, 256) global avg pool


class MoEFeatureExtractor(nn.Module):
    """MoEDualBranchEncoder → h_out (z_inv cat z_spec) → 64-dim feature."""
    def __init__(self, moe: MoEDualBranchEncoder):
        super().__init__()
        self.moe = moe
        self.feature_dim = moe.projection_output * 2   # 64

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h_out, _, _ = self.moe(x)
        return h_out


# ── Checkpoint loading ────────────────────────────────────────────────────────

def load_byol_encoder(ckpt_path: str) -> CNNFeatureExtractor:
    cnn  = CNNEncoder(**CNN_ARGS)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd   = ckpt.get("state_dict", ckpt)
    enc_sd = {k[len("online_encoder."):]: v
              for k, v in sd.items() if k.startswith("online_encoder.")}
    msg = cnn.load_state_dict(enc_sd, strict=False)
    if msg.missing_keys:
        print(f"  [BYOL] missing keys: {msg.missing_keys}")
    print(f"  Loaded BYOL encoder from {ckpt_path}")
    return CNNFeatureExtractor(cnn)


def load_simsiam_encoder(ckpt_path: str) -> CNNFeatureExtractor:
    cnn  = CNNEncoder(**CNN_ARGS)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd   = ckpt.get("state_dict", ckpt)
    enc_sd = {k[len("encoder."):]: v
              for k, v in sd.items() if k.startswith("encoder.")}
    msg = cnn.load_state_dict(enc_sd, strict=False)
    if msg.missing_keys:
        print(f"  [SimSiam] missing keys: {msg.missing_keys}")
    print(f"  Loaded SimSiam encoder from {ckpt_path}")
    return CNNFeatureExtractor(cnn)


def load_moe_encoder(ckpt_path: str) -> MoEFeatureExtractor:
    moe  = MoEDualBranchEncoder(**MOE_ARGS)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd   = ckpt.get("state_dict", ckpt)
    enc_sd = {k[len("encoder."):]: v
              for k, v in sd.items() if k.startswith("encoder.")}
    msg = moe.load_state_dict(enc_sd, strict=False)
    if msg.missing_keys:
        print(f"  [MoE] missing keys: {msg.missing_keys}")
    print(f"  Loaded MoE encoder from {ckpt_path}")
    return MoEFeatureExtractor(moe)


LOADER_FNS = {
    "byol":                 load_byol_encoder,
    "simsiam":              load_simsiam_encoder,
    "dual_branch_no_scale": load_moe_encoder,
}


# ── Dataset helpers ───────────────────────────────────────────────────────────

def make_dataset(split: str, split_file: str, sub_sample_frac=None) -> WESADDataset:
    return WESADDataset(
        dataset_path    = WESAD_DATA,
        include_labels  = True,
        split           = split,
        split_file      = split_file,
        sub_sample_frac = sub_sample_frac,
    )


def make_loader(ds, batch_size, shuffle) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=FINETUNE_WORKERS, pin_memory=True, drop_last=False)


# ── Linear-probe training ─────────────────────────────────────────────────────

def train_linear_probe(extractor, train_loader, feature_dim):
    extractor = extractor.to(DEVICE).eval()
    for p in extractor.parameters():
        p.requires_grad_(False)

    head = nn.Linear(feature_dim, 1).to(DEVICE)
    opt  = torch.optim.Adam(head.parameters(), lr=FINETUNE_LR)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(FINETUNE_EPOCHS):
        head.train()
        for batch in train_loader:
            x = batch["x"].to(DEVICE).float()
            y = batch["y"].to(DEVICE).float().view(-1, 1)
            feat = extractor(x)
            loss = loss_fn(head(feat), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    return head


@torch.no_grad()
def predict(extractor, head, loader):
    extractor.eval()
    head.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        x = batch["x"].to(DEVICE).float()
        y = batch["y"].numpy().flatten().astype(int)
        feat  = extractor(x)
        logit = head(feat).squeeze(-1)
        preds = (torch.sigmoid(logit) >= 0.5).long().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y)
    return np.concatenate(all_preds), np.concatenate(all_labels)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_model(model_name: str, ckpt_path: str):
    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")

    fold_files = sorted(f for f in os.listdir(LOSO_DIR) if f.endswith(".csv"))
    all_preds, all_labels, all_folds = [], [], []

    for fold_idx, fold_file in enumerate(fold_files):
        split_file = os.path.join(LOSO_DIR, fold_file)
        print(f"\n  Fold {fold_idx:02d} / {len(fold_files)-1}  ({fold_file})")

        # fresh encoder per fold (no leakage across folds)
        extractor = LOADER_FNS[model_name](ckpt_path).to(DEVICE)
        feature_dim = extractor.feature_dim

        train_ds = make_dataset("train", split_file, sub_sample_frac=SUBSAMPLE_FRAC)
        test_ds  = make_dataset("test",  split_file, sub_sample_frac=None)

        train_loader = make_loader(train_ds, FINETUNE_BATCH, shuffle=True)
        test_loader  = make_loader(test_ds,  FINETUNE_BATCH, shuffle=False)

        head = train_linear_probe(extractor, train_loader, feature_dim)
        y_pred, y_true = predict(extractor, head, test_loader)

        fold_f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
        fold_acc = accuracy_score(y_true, y_pred)
        print(f"    F1={fold_f1:.4f}  Acc={fold_acc:.4f}  n_test={len(y_true)}")

        all_preds.append(y_pred)
        all_labels.append(y_true)
        all_folds.append(np.full(len(y_true), fold_idx, dtype=int))

    results = {
        "model":    model_name,
        "y_pred":   np.concatenate(all_preds),
        "y_true":   np.concatenate(all_labels),
        "fold_id":  np.concatenate(all_folds),
    }

    out_f1  = f1_score(results["y_true"], results["y_pred"], average="macro", zero_division=0)
    out_acc = accuracy_score(results["y_true"], results["y_pred"])
    print(f"\n  Overall — F1={out_f1:.4f}  Acc={out_acc:.4f}  n={len(results['y_true'])}")

    save_path = os.path.join(SAVE_DIR, f"psychionet_{model_name}_preds.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(results, f)
    print(f"  Saved → {save_path}")
    return results


def main():
    print(f"Device: {DEVICE}")
    print(f"LOSO folds: {LOSO_DIR}")

    all_results = {}
    for model_name, ckpt_path in CHECKPOINTS.items():
        if not os.path.isfile(ckpt_path):
            print(f"WARNING: checkpoint not found: {ckpt_path}")
            continue
        all_results[model_name] = run_model(model_name, ckpt_path)

    print("\n\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"  {'Model':<30}  {'F1':>8}  {'Acc':>8}")
    print(f"  {'-'*50}")
    for name, res in all_results.items():
        f1  = f1_score(res["y_true"], res["y_pred"], average="macro", zero_division=0)
        acc = accuracy_score(res["y_true"], res["y_pred"])
        print(f"  {name:<30}  {f1:>8.4f}  {acc:>8.4f}")
    print(f"\nPredictions saved to: {SAVE_DIR}")
    print("Run bootstrap_psychionet.ipynb next to compute pairwise statistics.")


if __name__ == "__main__":
    main()
