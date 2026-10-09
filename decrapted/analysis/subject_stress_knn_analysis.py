"""
Quantitative validation of the "subject-specific branch encodes identity, not
task-relevant stress information" claim.

For each pretrained MoE dual-branch checkpoint, we extract per-sample
embeddings from a labelled downstream dataset (subject id + stress label are
both available), and run a leave-one-out k-NN analysis in each head's
representation space:

  - same-subject purity@k : avg. fraction of a sample's k nearest neighbours
                             that belong to the SAME subject
  - same-stress purity@k  : avg. fraction of a sample's k nearest neighbours
                             that share the SAME stress label
  - k-NN accuracy          : majority-vote classification accuracy using the
                             k nearest neighbours as the classifier

Each is reported against its empirical chance baseline (the probability that
two randomly paired samples would share the same subject / stress label,
given the dataset's actual group sizes / class balance).

If the subject-specific head (z_spec) organizes representations by subject
identity rather than stress, we expect:
  same-subject purity/accuracy >> chance,  well above same-stress purity/accuracy
in z_spec, with the opposite (or much weaker) pattern in the invariant head
(z_inv), which is what the contrastive objective explicitly optimizes for.

Usage:
    python analysis/subject_stress_knn_analysis.py
    python analysis/subject_stress_knn_analysis.py --k 1 5 10 20 --device cuda
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datasets.wesad_dataset import WESADDataset
from datasets.swell_stressid_dataset import SWELL_STRESSID_Dataset
from models.net.moe_encoder import MoEDualBranchEncoder


# ─────────────────────────────────────────────────────────────────────────
#  Checkpoints to evaluate: (run_name, pretrain_dir, eval_dataset_name)
#  pretrain_dir must contain `pretrain/config.json` and `pretrain/encoder_best_.pt`
# ─────────────────────────────────────────────────────────────────────────
RUNS = [
    # ── in-domain (pretrain and eval on the same dataset) — the cleanest test
    #    of the architecture's own disentanglement behaviour, no cross-dataset
    #    domain-shift confound ──────────────────────────────────────────────
    ("WESAD-pretrained -> eval WESAD",
     os.path.join(REPO_ROOT, "save/dual_branch_no_scale_new/moe_dual_branch_WESADDataset_0"),
     "WESADDataset"),
    ("SWELL-pretrained -> eval SWELL",
     os.path.join(REPO_ROOT, "save/dual_branch_no_scale_new/moe_dual_branch_SWELLDataset_0"),
     "SWELLDataset"),
    # ── cross-dataset transfer (matches the paper's finetune protocol) —
    #    kept for context, but domain shift can itself explain head asymmetries,
    #    so don't treat these as clean evidence about the architecture ───────
    ("SWELL-pretrained -> eval WESAD",
     os.path.join(REPO_ROOT, "save/dual_branch_no_scale_new/moe_dual_branch_SWELLDataset_0"),
     "WESADDataset"),
    ("PsychioNet-pretrained -> eval WESAD",
     os.path.join(REPO_ROOT, "save/dual_branch_no_scale_new/moe_dual_branch_PsychioNet_0"),
     "WESADDataset"),
    ("PsychioNet-pretrained -> eval SWELL",
     os.path.join(REPO_ROOT, "save/dual_branch_no_scale_new/moe_dual_branch_PsychioNet_0"),
     "SWELLDataset"),
]

EVAL_DATASETS = {
    "WESADDataset": "/home/s223149341/SSL-invariance-Subject_Project_model/data/WESAD/wesad_10_05_no_standardize",
    "SWELLDataset": "/home/s223149341/SSL-invariance-Subject_Project_model/data/SWELL/SWELL_1280_320_3label",
}

# segment_length // segment_stride (data_preparation/config.py) for each dataset.
# Consecutive windows overlap by (ratio-1)/ratio of their length (e.g. WESAD's
# 1280-sample windows / 64-sample stride overlap 95%), so raw adjacent samples
# are near-duplicates. Left un-deduplicated, k-NN "purity" mostly rediscovers
# this temporal redundancy (near-identical neighbours) rather than measuring
# what the encoder learned. We keep only every `ratio`-th window per subject
# (in on-disk/temporal order) so kept windows do not overlap at all.
NONOVERLAP_RATIO = {
    "WESADDataset": 20,  # 1280 / 64
    "SWELLDataset": 4,   # 1280 / 320
}

HEADS = ["z_inv", "z_spec", "h_out"]  # z_inv/h_out included as contrast to z_spec

BATCH_SIZE = 256
CHUNK_SIZE = 2000  # queries processed per k-NN similarity chunk


# ─────────────────────────────────────────────────────────────────────────
#  Encoder / dataset loading
# ─────────────────────────────────────────────────────────────────────────
def load_encoder(pretrain_dir, device):
    with open(os.path.join(pretrain_dir, "pretrain", "config.json")) as f:
        cfg = json.load(f)
    enc_args = cfg["pretrain_args"]["model_args"]["base_encoder_args"]

    encoder = MoEDualBranchEncoder(
        input_dim=enc_args["input_dim"],
        dropout_prob=0.0,
        kernel_size=enc_args["kernel_size"],
        stride=enc_args["stride"],
        output_dim=enc_args["output_dim"],
        projection_output=enc_args["projection_output"],
    ).to(device)

    ckpt_path = os.path.join(pretrain_dir, "pretrain", "encoder_best_.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    enc_sd = {}
    for k, v in sd.items():
        k = k.replace("module.", "", 1)
        for prefix in ("encoder_q.", "encoder.", "encoder_i."):
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        enc_sd[k] = v

    msg = encoder.load_state_dict(enc_sd, strict=False)
    if msg.missing_keys:
        print(f"  [warn] missing keys when loading {ckpt_path}: {msg.missing_keys}")
    encoder.eval()
    return encoder


def load_eval_dataset(dataset_name, dedup_overlap=True):
    dataset_path = EVAL_DATASETS[dataset_name]
    ds_args = dict(
        dataset_path=dataset_path,
        include_labels=True,
        split=None,
        sub_sample_frac=None,
        data_views=None,
        transform_dict_core=None,
        transform_dict_artifact=None,
        transform_global_core=None,
        transform_global_artifact=None,
    )
    if dataset_name == "WESADDataset":
        ds = WESADDataset(**ds_args)
    elif dataset_name == "SWELLDataset":
        ds = SWELL_STRESSID_Dataset(**ds_args)
    else:
        raise ValueError(dataset_name)

    if dedup_overlap:
        ratio = NONOVERLAP_RATIO[dataset_name]
        # rows are stored in contiguous per-subject, chronologically-ordered
        # blocks; keeping every `ratio`-th row per subject removes all
        # window overlap without needing timestamps.
        subj_ids = np.array([r["subject_id_int"] for r in ds.data_list])
        keep = []
        for s in np.unique(subj_ids):
            idx = np.flatnonzero(subj_ids == s)
            keep.append(idx[::ratio])
        keep = np.sort(np.concatenate(keep))
        print(f"  De-duplicating overlapping windows (keep every {ratio}th/subject): "
              f"{len(ds)} -> {len(keep)} samples")
        ds = Subset(ds, keep)

    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                       pin_memory=torch.cuda.is_available(), drop_last=False)


@torch.no_grad()
def extract_embeddings(encoder, loader, device):
    heads = {"z_inv": [], "z_spec": [], "h_out": []}
    subj_list, label_list = [], []

    for batch in loader:
        x = batch["x"].to(device).float()
        _h, h_out, _z, z_inv, z_spec = encoder(x, return_embeddings=True)
        heads["z_inv"].append(z_inv.cpu().numpy())
        heads["z_spec"].append(z_spec.cpu().numpy())
        heads["h_out"].append(h_out.cpu().numpy())
        subj_list.append(batch["subject_id_int"].cpu().numpy())
        label_list.append(batch["y"].cpu().numpy().flatten().astype(int))

    out = {name: np.vstack(vals) for name, vals in heads.items()}
    out["subj"] = np.concatenate(subj_list)
    out["label"] = np.concatenate(label_list)
    return out


# ─────────────────────────────────────────────────────────────────────────
#  Leave-one-out k-NN purity / accuracy
# ─────────────────────────────────────────────────────────────────────────
def empirical_chance_purity(group_ids):
    """P(two distinct, randomly paired samples share the same group id)."""
    _, counts = np.unique(group_ids, return_counts=True)
    n = len(group_ids)
    same_pairs = np.sum(counts * (counts - 1))
    total_pairs = n * (n - 1)
    return same_pairs / total_pairs


def majority_class_baseline(labels):
    _, counts = np.unique(labels, return_counts=True)
    return counts.max() / counts.sum()


def knn_purity_and_accuracy(embeddings, subj, label, k_list, device):
    """
    For every sample (query), find its k nearest neighbours (cosine
    similarity, self excluded) among all OTHER samples, then compute:
      - purity: fraction of the k neighbours sharing the query's subject / label
      - accuracy: majority-vote prediction of subject / label vs. ground truth
    """
    X = torch.from_numpy(embeddings).float().to(device)
    X = torch.nn.functional.normalize(X, dim=1)
    subj_t = torch.from_numpy(subj).long().to(device)
    label_t = torch.from_numpy(label).long().to(device)

    n = X.shape[0]
    max_k = max(k_list)

    subj_purity_sum = {k: 0.0 for k in k_list}
    label_purity_sum = {k: 0.0 for k in k_list}
    subj_correct = {k: 0 for k in k_list}
    label_correct = {k: 0 for k in k_list}

    for start in range(0, n, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n)
        sims = X[start:end] @ X.T  # [chunk, n]
        # exclude self
        rows = torch.arange(end - start, device=device)
        sims[rows, torch.arange(start, end, device=device)] = -float("inf")

        top_sim, top_idx = torch.topk(sims, k=max_k, dim=1)  # [chunk, max_k]
        nbr_subj = subj_t[top_idx]     # [chunk, max_k]
        nbr_label = label_t[top_idx]   # [chunk, max_k]

        q_subj = subj_t[start:end].unsqueeze(1)
        q_label = label_t[start:end].unsqueeze(1)

        for k in k_list:
            subj_match_k = (nbr_subj[:, :k] == q_subj)
            label_match_k = (nbr_label[:, :k] == q_label)

            subj_purity_sum[k] += subj_match_k.float().sum().item()
            label_purity_sum[k] += label_match_k.float().sum().item()

            subj_pred = torch.mode(nbr_subj[:, :k], dim=1).values
            label_pred = torch.mode(nbr_label[:, :k], dim=1).values
            subj_correct[k] += (subj_pred == q_subj.squeeze(1)).sum().item()
            label_correct[k] += (label_pred == q_label.squeeze(1)).sum().item()

    results = []
    for k in k_list:
        results.append({
            "k": k,
            "same_subject_purity": subj_purity_sum[k] / (n * k),
            "same_stress_purity": label_purity_sum[k] / (n * k),
            "subject_knn_accuracy": subj_correct[k] / n,
            "stress_knn_accuracy": label_correct[k] / n,
        })
    return results


# ─────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_csv", type=str,
                         default=os.path.join(REPO_ROOT, "analysis", "subject_stress_knn_results.csv"))
    parser.add_argument("--keep_overlap", action="store_true",
                         help="Do NOT de-duplicate overlapping sliding-window samples "
                              "(purity will be inflated by temporally-adjacent near-duplicates).")
    args = parser.parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}")

    all_rows = []

    for run_name, pretrain_dir, eval_dataset_name in RUNS:
        print(f"\n=== {run_name} ===")
        encoder = load_encoder(pretrain_dir, device)
        loader = load_eval_dataset(eval_dataset_name, dedup_overlap=not args.keep_overlap)
        embeddings = extract_embeddings(encoder, loader, device)

        subj, label = embeddings["subj"], embeddings["label"]
        chance_subject = empirical_chance_purity(subj)
        chance_stress = max(empirical_chance_purity(label), majority_class_baseline(label))
        print(f"  N={len(subj)}  subjects={len(np.unique(subj))}  "
              f"chance(same-subject)={chance_subject:.4f}  chance(same-stress)={chance_stress:.4f}")

        for head in HEADS:
            print(f"  Head: {head}")
            res = knn_purity_and_accuracy(embeddings[head], subj, label, args.k, device)
            for r in res:
                r.update({
                    "run": run_name,
                    "eval_dataset": eval_dataset_name,
                    "head": head,
                    "chance_same_subject": chance_subject,
                    "chance_same_stress": chance_stress,
                })
                all_rows.append(r)
                print(f"    k={r['k']:>3}  "
                      f"same-subj purity={r['same_subject_purity']:.3f} (chance {chance_subject:.3f})  "
                      f"same-stress purity={r['same_stress_purity']:.3f} (chance {chance_stress:.3f})  "
                      f"subj-kNN acc={r['subject_knn_accuracy']:.3f}  "
                      f"stress-kNN acc={r['stress_knn_accuracy']:.3f}")

    df = pd.DataFrame(all_rows)
    cols = ["run", "eval_dataset", "head", "k",
            "same_subject_purity", "chance_same_subject",
            "same_stress_purity", "chance_same_stress",
            "subject_knn_accuracy", "stress_knn_accuracy"]
    df = df[cols]
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved results -> {args.out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
