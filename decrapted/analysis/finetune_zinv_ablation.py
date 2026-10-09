"""
z_inv-only fine-tuning ablation.

For each of a set of ALREADY-COMPLETED h_out finetune runs (existing
results.csv under save/dual_branch_no_scale_new/*/finetune/<eval_dataset>/),
re-run fine-tuning with the SAME pretrained encoder checkpoint, the SAME
LOSO fold splits, and the SAME optimizer hyperparameters -- but with a
classifier that reads only z_inv (models/moe_dual_branch_zinv_only.py),
dropping z_spec from the classifier input entirely. Compares the resulting
per-fold F1/accuracy against the existing h_out baseline, to test whether
z_spec's presence in the classifier input helps, hurts, or makes no
difference to downstream stress classification.

Usage:
    python analysis/finetune_zinv_ablation.py --run psychionet_wesad
    python analysis/finetune_zinv_ablation.py --run psychionet_wesad wesad_swell psychionet_swell
    python analysis/finetune_zinv_ablation.py --run psychionet_wesad --max_folds 1 --epochs 2   # smoke test
"""
import argparse
import copy
import json
import os
import sys

import pandas as pd
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)  # model_path in saved configs is relative to repo root

from utils import setup_logger
from trainer.finetuner import Finetuner
from trainer.utils import get_loss
from models.utils import get_base_encoder
from models.moe_dual_branch_zinv_only import MoEFinetuneModelZInvOnly


# ─────────────────────────────────────────────────────────────────────────
#  Runs to ablate. `has_baseline=True` means an h_out finetune already ran
#  to completion (existing results.csv) and is reused as-is; `False` means
#  no completed h_out run exists for this combo, so both arms (h_out and
#  z_inv-only) are trained fresh here, from the same pretrained checkpoint,
#  same LOSO folds, same hyperparameters -- for a fair, apples-to-apples
#  comparison.
# ─────────────────────────────────────────────────────────────────────────
RUN_CONFIGS = {
    "wesad_wesad": {
        "dir": os.path.join(
            REPO_ROOT, "save/dual_branch_no_scale_new/moe_dual_branch_WESADDataset_0/finetune/WESADDataset"),
        "has_baseline": False,  # only fold-0 config.json exists, no completed results.csv
    },
    "psychionet_wesad": {
        "dir": os.path.join(
            REPO_ROOT, "save/dual_branch_no_scale_new/moe_dual_branch_PsychioNet_0/finetune/WESADDataset"),
        "has_baseline": True,
    },
    "wesad_swell": {
        "dir": os.path.join(
            REPO_ROOT, "save/dual_branch_no_scale_new/moe_dual_branch_WESADDataset_0/finetune/SWELLDataset"),
        "has_baseline": True,
    },
    "psychionet_swell": {
        "dir": os.path.join(
            REPO_ROOT, "save/dual_branch_no_scale_new/moe_dual_branch_PsychioNet_0/finetune/SWELLDataset"),
        "has_baseline": True,
    },
}

OUTPUT_ROOT = os.path.join(REPO_ROOT, "analysis", "zinv_only_ablation")


class FairFinetuner(Finetuner):
    """
    Identical to Finetuner, with one fix: rebuild train_loader with
    drop_last=True. Under world_size=1, these small (1%-subsampled) LOSO
    folds can leave a final batch of size 1, which crashes BatchNorm in
    training mode. val/test loaders are untouched (dropping eval samples
    would distort reported metrics). Used as the common base for both arms
    of a from-scratch comparison (wesad_wesad) so the only difference
    between them is the classifier input, not an incidental data-loading
    discrepancy.
    """

    def _build_dataloader(self):
        super()._build_dataloader()
        self.train_loader = self._make_loader(
            self.train_loader.dataset, shuffle=False,
            sampler=self.train_sampler, drop_last=True,
        )


class ZInvOnlyFinetuner(FairFinetuner):
    """Identical to FairFinetuner, except the fine-tune model is z_inv-only."""

    def _build_model(self, training_cfg):
        loss_fn = get_loss(training_cfg["loss"]["name"], training_cfg["loss"]["loss_args"])
        enc_args = training_cfg["model_args"]["base_encoder_args"]
        encoder = get_base_encoder(training_cfg["model_args"]["base_encoder"], enc_args)
        self.model = MoEFinetuneModelZInvOnly(
            moe_encoder=encoder,
            num_class=training_cfg["model_args"].get("num_class", 1),
            model_path=training_cfg["model_args"].get("model_path", None),
            device=self.device,
        )
        self.model.set_loss_fn(loss_fn)
        self.model.to(self.device)


def load_baseline_results(baseline_dir):
    df = pd.read_csv(os.path.join(baseline_dir, "results.csv"))
    return df.reset_index().rename(columns={"index": "fold"})


def run_ablation(run_key, device, epochs_override=None, max_folds=None, resume=False):
    info = RUN_CONFIGS[run_key]
    cfg_dir = info["dir"]
    has_baseline = info["has_baseline"]
    with open(os.path.join(cfg_dir, "config.json")) as f:
        cfg = json.load(f)

    split_fold = cfg["split_path"]
    folds = sorted(p for p in os.listdir(split_fold) if p.endswith(".csv"))

    baseline_df = None
    if has_baseline:
        baseline_df = load_baseline_results(cfg_dir)
        assert len(folds) == len(baseline_df), \
            f"fold count mismatch: {len(folds)} split files vs {len(baseline_df)} baseline rows"
        print(f"[{run_key}] reusing completed h_out baseline: {cfg_dir}  ({len(baseline_df)} folds)")
    else:
        print(f"[{run_key}] no completed h_out baseline found -- training h_out AND z_inv-only "
              f"both fresh, from {cfg_dir}  ({len(folds)} folds)")

    if max_folds is not None:
        folds = folds[:max_folds]

    out_dir = os.path.join(OUTPUT_ROOT, run_key)
    hout_dir = os.path.join(out_dir, "h_out_rerun")   # only used when has_baseline=False
    zinv_dir = os.path.join(out_dir, "zinv_only")
    os.makedirs(hout_dir, exist_ok=True)
    os.makedirs(zinv_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "results.csv")

    ablation_rows = []
    start_fold = 0
    if resume and os.path.exists(results_path):
        existing = pd.read_csv(results_path)
        existing = existing[existing["fold"] < len(folds)]  # ignore stale rows beyond current fold count
        ablation_rows = existing.to_dict("records")
        start_fold = len(ablation_rows)
        print(f"[{run_key}] resuming: {start_fold} fold(s) already completed, continuing from fold {start_fold}")

    logger = setup_logger(out_dir, name=f"zinv_ablation_{run_key}")

    if epochs_override is not None:
        cfg["finetune_args"]["optim_args"]["epochs"] = epochs_override

    # These per-fold datasets are tiny (LOSO + 1% subsample); num_workers=8
    # spawns 8 worker processes every epoch, which massively dominates
    # runtime for datasets this small. num_workers is purely a DataLoader
    # performance knob (no effect on training dynamics/results given the
    # sampler already handles shuffling), so this doesn't affect comparability.
    cfg["finetune_args"]["optim_args"]["num_workers"] = 0

    for run_id, fold_file in enumerate(folds):
        if run_id < start_fold:
            continue  # already completed in a previous (resumed) invocation
        split_file = os.path.join(split_fold, fold_file)

        if has_baseline:
            h_out_f1 = baseline_df.loc[run_id, "best_f1"]
            h_out_acc = baseline_df.loc[run_id, "best_acc"]
        else:
            hout_cfg = copy.deepcopy(cfg)
            for split in ["train_dataset_args", "val_dataset_args", "test_dataset_args"]:
                hout_cfg["finetune_args"]["dataset_args"][split]["split_file"] = split_file
            hout_cfg["logging_args"]["finetune_output_dir"] = hout_dir
            hout_finetuner = FairFinetuner(
                hout_cfg, logger=logger, device=device, rank=0, world_size=1, seed=0, fold=run_id,
            )
            hout_out = hout_finetuner.train()
            h_out_f1, h_out_acc = hout_out["best_f1"], hout_out["best_acc"]

        zinv_cfg = copy.deepcopy(cfg)
        for split in ["train_dataset_args", "val_dataset_args", "test_dataset_args"]:
            zinv_cfg["finetune_args"]["dataset_args"][split]["split_file"] = split_file
        zinv_cfg["logging_args"]["finetune_output_dir"] = zinv_dir

        finetuner = ZInvOnlyFinetuner(
            zinv_cfg, logger=logger, device=device, rank=0, world_size=1, seed=0, fold=run_id,
        )
        out = finetuner.train()
        print(f"  fold {run_id}: z_inv-only F1={out['best_f1']:.4f} acc={out['best_acc']:.4f}"
              f"  |  h_out F1={h_out_f1:.4f} acc={h_out_acc:.4f}"
              f"{'  (reused baseline)' if has_baseline else '  (retrained here)'}")
        ablation_rows.append({
            "fold": run_id,
            "zinv_only_f1": out["best_f1"],
            "zinv_only_acc": out["best_acc"],
            "h_out_f1": h_out_f1,
            "h_out_acc": h_out_acc,
        })
        pd.DataFrame(ablation_rows).to_csv(results_path, index=False)  # checkpoint after every fold

    df = pd.DataFrame(ablation_rows)
    df["f1_delta_zinv_minus_hout"] = df["zinv_only_f1"] - df["h_out_f1"]
    df["acc_delta_zinv_minus_hout"] = df["zinv_only_acc"] - df["h_out_acc"]
    df.to_csv(results_path, index=False)

    print(f"\n[{run_key}] SUMMARY (n={len(df)} folds)")
    print(f"  h_out      : F1={df['h_out_f1'].mean():.4f} +/- {df['h_out_f1'].std():.4f}"
          f"   acc={df['h_out_acc'].mean():.4f} +/- {df['h_out_acc'].std():.4f}")
    print(f"  z_inv-only : F1={df['zinv_only_f1'].mean():.4f} +/- {df['zinv_only_f1'].std():.4f}"
          f"   acc={df['zinv_only_acc'].mean():.4f} +/- {df['zinv_only_acc'].std():.4f}")
    print(f"  mean delta (z_inv-only - h_out): F1={df['f1_delta_zinv_minus_hout'].mean():+.4f}"
          f"  acc={df['acc_delta_zinv_minus_hout'].mean():+.4f}")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, nargs="+", default=list(RUN_CONFIGS.keys()),
                         choices=list(RUN_CONFIGS.keys()))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None,
                         help="override epochs/fold (default: whatever the baseline config used, 100)")
    parser.add_argument("--max_folds", type=int, default=None, help="limit number of folds (for smoke tests)")
    parser.add_argument("--resume", action="store_true",
                         help="skip folds already present in analysis/zinv_only_ablation/<run>/results.csv")
    args = parser.parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}")

    all_summaries = {}
    for run_key in args.run:
        df = run_ablation(run_key, device, epochs_override=args.epochs,
                           max_folds=args.max_folds, resume=args.resume)
        all_summaries[run_key] = df

    print("\n" + "=" * 70)
    print("FINAL SUMMARY ACROSS RUNS")
    print("=" * 70)
    for run_key, df in all_summaries.items():
        print(f"{run_key}: h_out F1={df['h_out_f1'].mean():.4f}  "
              f"z_inv-only F1={df['zinv_only_f1'].mean():.4f}  "
              f"delta={df['f1_delta_zinv_minus_hout'].mean():+.4f}")


if __name__ == "__main__":
    main()
