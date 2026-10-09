"""
Paired bootstrap significance test (n=1000) for the SASContrast vs. SimSiam
comparisons in Table 2, requested during review.

Unit of resampling: LOSO fold (= held-out subject), the same unit Table 2's
mean F1/Accuracy is already averaged over. For each of the 1000 draws, folds
are resampled with replacement and the mean F1/Accuracy difference
(SASContrast - SimSiam) is recomputed; the resulting distribution gives a
95% CI and a two-sided p-value for whether the true mean difference is 0.

Per-fold predictions are read directly from the exported finetuned_best_{i}.csv
files (label, pred columns) rather than from results.csv, so scores are
recomputed here rather than trusted from the logged summary.

Data-integrity notes (verified by cross-checking every fold's recomputed
F1/Accuracy against the corresponding row logged in results.csv):
  - 7 of 8 run directories: every fold's on-disk prediction file matches the
    logged Table 2 score exactly.
  - simsiam_SWELLDataset_0/finetune/WESADDataset: fold 0's on-disk file
    (F1=0.4054) does NOT match the logged score for that fold (F1=0.5635).
    The file was overwritten by a later, unlogged rerun. This fold is
    excluded from the SWELL-pretrained -> WESAD comparison for BOTH arms
    (n=14 instead of 15) to keep the pairing intact.
  - An instance-level (row-order) pairing was also attempted but abandoned:
    spot checks show prediction files for the same fold index are not always
    written in the same row order across different runs/models (label
    multiset matches, row-by-row order does not), so per-row alignment across
    models cannot be assumed in general. Only fold-level aggregate metrics
    (which do not depend on row order within a file) are used.
"""
import csv
import json
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

N_BOOT = 1000


def load_fold(path):
    rows = list(csv.reader(open(path)))[1:]
    y_true = np.array([int(r[0]) for r in rows])
    y_pred = np.array([int(r[1]) for r in rows])
    return y_true, y_pred


def load_run(finetune_dir, n_folds, exclude=(), updated_folds=()):
    """updated_folds: fold indices to read from finetuned_best_{i}_updated.csv
    instead of finetuned_best_{i}.csv (used when a subset of folds was
    rerun after the original run and the fix wrote *_updated.csv siblings
    instead of overwriting the originals)."""
    out = []
    for i in range(n_folds):
        if i in exclude:
            continue
        suffix = "_updated" if i in updated_folds else ""
        out.append(load_fold(f"{finetune_dir}/finetuned_best_{i}{suffix}.csv"))
    return out


def fold_metrics(finetune_dir, n_folds, exclude=(), updated_folds=()):
    f1s, accs = [], []
    for i in range(n_folds):
        if i in exclude:
            continue
        suffix = "_updated" if i in updated_folds else ""
        y_true, y_pred = load_fold(f"{finetune_dir}/finetuned_best_{i}{suffix}.csv")
        f1s.append(f1_score(y_true, y_pred, average="macro", zero_division=0))
        accs.append(accuracy_score(y_true, y_pred))
    return np.array(f1s), np.array(accs)


def paired_fold_bootstrap(a, b, n_boot=N_BOOT, seed=0):
    """Resamples the n already-aggregated per-fold F1/Accuracy scores.
    Does not touch individual sample predictions."""
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[k] = a[idx].mean() - b[idx].mean()
    return diffs


def sample_level_bootstrap(folds_a, folds_b, n_boot=N_BOOT, seed=0):
    """Two-stage cluster bootstrap that resamples actual per-sample
    predictions, not just the pre-aggregated per-fold scores.

    Stage 1: resample fold ids (shared between A and B -- valid, since fold
    index i is the same held-out subject for both models).
    Stage 2: within each resampled fold, resample instances *independently*
    for A and B (NOT the same row indices) -- required because row order is
    not guaranteed aligned between the two models' CSVs for a given fold
    (confirmed: same label multiset per fold, different row order in at
    least one setting). Each model's resample stays internally consistent
    (label_i still pairs with pred_i within that model's own array); no
    cross-model row alignment is assumed.
    """
    rng = np.random.default_rng(seed)
    n_folds = len(folds_a)
    diffs_f1 = np.empty(n_boot)
    diffs_acc = np.empty(n_boot)
    for b in range(n_boot):
        fold_ids = rng.integers(0, n_folds, size=n_folds)
        f1_a_list, f1_b_list, acc_a_list, acc_b_list = [], [], [], []
        for fid in fold_ids:
            yt_a, pred_a = folds_a[fid]
            yt_b, pred_b = folds_b[fid]

            idx_a = rng.integers(0, len(yt_a), size=len(yt_a))
            idx_b = rng.integers(0, len(yt_b), size=len(yt_b))

            f1_a_list.append(f1_score(yt_a[idx_a], pred_a[idx_a], average="macro", zero_division=0))
            f1_b_list.append(f1_score(yt_b[idx_b], pred_b[idx_b], average="macro", zero_division=0))
            acc_a_list.append(accuracy_score(yt_a[idx_a], pred_a[idx_a]))
            acc_b_list.append(accuracy_score(yt_b[idx_b], pred_b[idx_b]))
        diffs_f1[b] = np.mean(f1_a_list) - np.mean(f1_b_list)
        diffs_acc[b] = np.mean(acc_a_list) - np.mean(acc_b_list)
    return diffs_f1, diffs_acc


def summarize(diffs, observed):
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_le, p_ge = np.mean(diffs <= 0), np.mean(diffs >= 0)
    p_value = min(2 * min(p_le, p_ge), 1.0)
    return {
        "observed_diff": float(observed),
        "ci_2.5": float(lo),
        "ci_97.5": float(hi),
        "p_value": float(p_value),
        "significant_0.05": bool(not (lo <= 0 <= hi)),
    }


SETTINGS = [
    {
        "name": "PsychioNet-pretrained -> WESAD finetune",
        "n_folds": 15,
        "a_dir": "./save/dual_branch_no_scale_new/moe_dual_branch_PsychioNet_0/finetune/WESADDataset",
        "b_dir": "./save/Simsiam/simsiam_PsychioNet_0/finetune/WESADDataset",
        "exclude": set(),
        # user-requested: both SASContrast and SimSiam now have updated
        # predictions (all 15 folds have finetuned_best_{i}_updated.csv
        # siblings, verified against each run's logged results.csv)
        "a_updated_folds": set(range(15)),
        "b_updated_folds": set(range(15)),
    },
    {
        "name": "PsychioNet-pretrained -> SWELL finetune",
        "n_folds": 25,
        "a_dir": "./save/dual_branch_no_scale_new/moe_dual_branch_PsychioNet_0/finetune/SWELLDataset",
        "b_dir": "./save/Simsiam/simsiam_PsychioNet_0/finetune/SWELLDataset",
        "exclude": set(),
        # user-requested: SimSiam PsychioNet->SWELL now has updated
        # predictions for all 25 folds, verified against results.csv
        "b_updated_folds": set(range(25)),
    },
    {
        "name": "WESAD-pretrained -> SWELL finetune",
        "n_folds": 25,
        "a_dir": "./save/dual_branch_no_scale_new/moe_dual_branch_WESADDataset_0/finetune/SWELLDataset",
        "b_dir": "./save/Simsiam/simsiam_WESADDataset_0/finetune/SWELLDataset",
        "exclude": set(),
    },
    {
        "name": "SWELL-pretrained -> WESAD finetune",
        "n_folds": 15,
        "a_dir": "./save/dual_branch_no_scale_new/moe_dual_branch_SWELLDataset_0/finetune/WESADDataset",
        "b_dir": "./save/Simsiam/simsiam_SWELLDataset_0/finetune/WESADDataset",
        # SimSiam fold 0's original on-disk file was corrupted (0.4054 vs
        # logged 0.5635 F1); the update below replaces it with a verified
        # fold 0 (F1=0.7487, matches results_updated.csv), so no exclusion
        # is needed anymore.
        "exclude": set(),
        # user-requested: both SASContrast and SimSiam now have updated
        # predictions (all 15 folds have finetuned_best_{i}_updated.csv
        # siblings, verified against each run's logged results)
        "a_updated_folds": set(range(15)),
        "b_updated_folds": set(range(15)),
    },
]

if __name__ == "__main__":
    results = []
    for setting_idx, s in enumerate(SETTINGS):
        a_updated = s.get("a_updated_folds", set())
        b_updated = s.get("b_updated_folds", set())
        folds_a = load_run(s["a_dir"], s["n_folds"], s["exclude"], updated_folds=a_updated)
        folds_b = load_run(s["b_dir"], s["n_folds"], s["exclude"], updated_folds=b_updated)
        f1_a, acc_a = fold_metrics(s["a_dir"], s["n_folds"], s["exclude"], updated_folds=a_updated)
        f1_b, acc_b = fold_metrics(s["b_dir"], s["n_folds"], s["exclude"], updated_folds=b_updated)

        seed = setting_idx  # fixed, deterministic seed for reproducibility
        diffs_f1_fold = paired_fold_bootstrap(f1_a, f1_b, seed=seed)
        diffs_acc_fold = paired_fold_bootstrap(acc_a, acc_b, seed=seed)
        diffs_f1_sample, diffs_acc_sample = sample_level_bootstrap(folds_a, folds_b, seed=seed)

        res = {
            "setting": s["name"],
            "n_folds": len(f1_a),
            "excluded_folds": sorted(s["exclude"]),
            "SASContrast_F1_mean": float(f1_a.mean()),
            "SimSiam_F1_mean": float(f1_b.mean()),
            "SASContrast_Acc_mean": float(acc_a.mean()),
            "SimSiam_Acc_mean": float(acc_b.mean()),
            "fold_level_bootstrap": {
                "F1": summarize(diffs_f1_fold, f1_a.mean() - f1_b.mean()),
                "Accuracy": summarize(diffs_acc_fold, acc_a.mean() - acc_b.mean()),
            },
            "sample_level_bootstrap": {
                "F1": summarize(diffs_f1_sample, f1_a.mean() - f1_b.mean()),
                "Accuracy": summarize(diffs_acc_sample, acc_a.mean() - acc_b.mean()),
            },
        }
        results.append(res)
        print(json.dumps(res, indent=2))

    with open("analysis/table2_bootstrap_results.json", "w") as f:
        json.dump(results, f, indent=2)
