#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# run_bootstrap_psychionet.sh
#
# Step 1: Generate per-sample predictions for BYOL, SimSiam, DualBranch-no-scale
#         (pretrained on PsychioNet, fine-tuned on WESAD LOSO).
# Step 2: Convert the bootstrap notebook to a script and run it.
#
# Usage:
#   cd subject_aware_contrastive_learning
#   bash scripts/run_bootstrap_psychionet.sh
# ──────────────────────────────────────────────────────────────────────────────

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "========================================================"
echo "  Step 1: Generate predictions (BYOL, SimSiam, MoE)"
echo "========================================================"
conda run -n pytorch310 python generate_predictions_psychionet.py

echo ""
echo "========================================================"
echo "  Step 2: Run bootstrap notebook"
echo "========================================================"
conda run -n pytorch310 jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=600 \
    bootstrap_psychionet.ipynb

echo ""
echo "Done. Results saved to save/predictions/"
echo "  bootstrap_results.csv     — overall pairwise p-values"
echo "  bootstrap_per_fold.csv    — per-fold pairwise p-values"
echo "  bootstrap_distributions.png"
echo "  bootstrap_heatmap.png"
echo "  per_fold_f1_boxplot.png"
