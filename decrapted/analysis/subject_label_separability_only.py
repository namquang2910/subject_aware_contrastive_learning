"""
Subject / label separability metric, on the de-duplicated (non-overlapping)
embeddings only -- reuses the same formula as the repo's existing
visualize_moe_umap_kmeans.ipynb / visualize_mode_with_random.ipynb notebooks:

    separability = mean(pairwise L2 dist, SAME group) / mean(pairwise L2 dist, DIFFERENT group)

computed on L2-normalized embeddings. Lower = tighter/more separated
clusters for that grouping (same-group points pulled closer together
relative to different-group points).

Reuses load_encoder / load_eval_dataset(dedup_overlap=True) / extract_embeddings
from analysis/subject_stress_knn_analysis.py, so this is on the identical
de-duplicated data as the rest of the disentanglement audit -- just this one
metric, nothing else (no UMAP, no clustering, no matched-pair test).

Usage:
    python analysis/subject_label_separability_only.py
"""
import os
import sys

import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from analysis.subject_stress_knn_analysis import RUNS, load_encoder, load_eval_dataset, extract_embeddings
from models.net.moe_encoder import MoEDualBranchEncoder
import json

HEADS = ["z_inv", "z_spec", "h_out"]
OUT_CSV = os.path.join(REPO_ROOT, "analysis", "subject_stress_disentanglement_plots", "separability_only.csv")


def calculate_label_separability(embeddings, labels):
    """
    Paper definition: ratio of mean DIFFERENT-label distance to mean
    SAME-label distance -- higher = better separation (unlike subject
    separability, which is same/diff and lower-is-better). This is the
    reciprocal of the repo's existing notebook implementation (which
    computes same/diff for label separability too, matching the
    subject-separability convention) -- that inherited version does not
    match the paper's stated formula/direction for this metric.
    """
    norm_embedding = F.normalize(embeddings, dim=1)
    distance = torch.cdist(norm_embedding, norm_embedding, p=2)
    same_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
    diff_mask = ~same_mask
    eye = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    same_mask = same_mask & ~eye
    return (distance[diff_mask].mean() / (distance[same_mask].mean() + 1e-8)).item()


def calculate_subject_separability(embeddings, subjects):
    norm_embedding = F.normalize(embeddings, dim=1)
    distance = torch.cdist(norm_embedding, norm_embedding, p=2)
    same_mask = subjects.unsqueeze(1) == subjects.unsqueeze(0)
    diff_mask = ~same_mask
    eye = torch.eye(subjects.size(0), dtype=torch.bool, device=subjects.device)
    same_mask = same_mask & ~eye
    return (distance[same_mask].mean() / (distance[diff_mask].mean() + 1e-8)).item()


def create_random_encoder(pretrain_dir, device):
    with open(os.path.join(pretrain_dir, "pretrain", "config.json")) as f:
        cfg = json.load(f)
    enc_args = cfg["pretrain_args"]["model_args"]["base_encoder_args"]
    encoder = MoEDualBranchEncoder(
        input_dim=enc_args["input_dim"], dropout_prob=0.0,
        kernel_size=enc_args["kernel_size"], stride=enc_args["stride"],
        output_dim=enc_args["output_dim"], projection_output=enc_args["projection_output"],
    ).to(device)
    encoder.eval()
    return encoder


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    rows = []
    for run_name, pretrain_dir, eval_dataset_name in RUNS:
        print(f"\n=== {run_name} ===")
        encoder = load_encoder(pretrain_dir, device)
        loader = load_eval_dataset(eval_dataset_name, dedup_overlap=True)
        emb = extract_embeddings(encoder, loader, device)

        rand_encoder = create_random_encoder(pretrain_dir, device)
        loader_rand = load_eval_dataset(eval_dataset_name, dedup_overlap=True)
        emb_rand = extract_embeddings(rand_encoder, loader_rand, device)

        subj = torch.from_numpy(emb["subj"]).long()
        label = torch.from_numpy(emb["label"]).long()

        for head in HEADS:
            X = torch.from_numpy(emb[head]).float()
            X_rand = torch.from_numpy(emb_rand[head]).float()
            subj_sep = calculate_subject_separability(X, subj)
            label_sep = calculate_label_separability(X, label)
            subj_sep_rand = calculate_subject_separability(X_rand, subj)
            label_sep_rand = calculate_label_separability(X_rand, label)
            print(f"  {head:8s}  subject_sep={subj_sep:.4f}  label_sep={label_sep:.4f}"
                  f"   (random-init: subject_sep={subj_sep_rand:.4f}  label_sep={label_sep_rand:.4f})")
            rows.append({
                "run": run_name, "eval_dataset": eval_dataset_name, "head": head,
                "subject_separability": subj_sep, "label_separability": label_sep,
                "subject_separability_random_init": subj_sep_rand,
                "label_separability_random_init": label_sep_rand,
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved -> {OUT_CSV}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
