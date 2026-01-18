"""
Utility functions to support model training
"""
import json
import time
import random
import os 
import torch
import numpy as np
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

import random
from torch.utils.data import Sampler

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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_timestamped_subdir(base_dir, exp_name="", dataset = ""):
    os.makedirs(base_dir, exist_ok=True)
    ts = time.strftime("%b_%d_%Y_%H-%M-%S", time.localtime())
    tag = f"{exp_name}_{dataset}_{ts}"
    path = os.path.join(base_dir, exp_name, tag)
    os.makedirs(path, exist_ok=True)
    return path

def save_config_file(config_dict, output_dir):
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)

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



def compute_metrics(y_true, y_hat):
    """
    Compute accuracy, precision, recall, F1-score, and confusion matrix.

    Args:
        y_true (Tensor or np.ndarray): Ground-truth labels.
        y_hat  (Tensor or np.ndarray): Predicted labels (class indices).

    Returns:
        dict: {
            'acc': float,
            'precision': float,
            'recall': float,
            'f1': float,
            'conf_mat': np.ndarray
        }
    """
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