"""
Utility functions to support model training
"""
import json
import time
import numpy as np
import logging
from torch.utils.data import Sampler
import random
import os 
import torch
import torch.distributed as dist
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import csv


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


def save_config_file(config_dict, output_dir):
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)
  

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


def broadcast_rank(obj, rank):
    if not dist.is_available() or not dist.is_initialized():
        return obj  # single-process fallback

    obj_list = [obj if rank == 0 else None]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]

def setup_distributed():
    """Initialize distributed training if WORLD_SIZE is set, else run single-process."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend="nccl")
        print(f"Distributed: rank {rank}/{world_size}, local_rank {local_rank}", flush=True)
    else:
        rank, local_rank = 0, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Single-process mode.", flush=True)

    return rank, world_size, device


def resolve_seeds(cfg: dict) -> list:
    if isinstance(cfg.get("seeds"), list) and cfg["seeds"]:
        return cfg["seeds"]
    if isinstance(cfg.get("seed"), int):
        return [cfg["seed"]]
    return [42]


def save_results(results: dict, file_path):
    headers, rows = [], []
    for k,v in results.items():
        headers.append(k)
        rows.append(v)        
    file_exists = os.path.isfile(file_path)
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(rows)
        
def create_experiment(base_dir, model_type = "", exp_name="", dataset="", mode="",finetune_dataset = None, seed=None, allow_exist=False):
    tag = f"{model_type}_{dataset}_{seed}" if model_type != "" or dataset != "" else ""
    out = os.path.join(base_dir, exp_name,tag, mode) if finetune_dataset is None else os.path.join(base_dir, exp_name,tag, mode, finetune_dataset)
    
    if os.path.exists(out):
        if allow_exist:
            print(f"Warning: path {out} already exists. Reusing.")
        else:
            print(f"Warning: path {out} already exists. Overwriting.")
            exit(1)
    else:
        os.makedirs(out, exist_ok=True)
    return out

def setup_logger(output_dir, name="train"):  # add a name param
    log_path = os.path.join(output_dir, "train.log")
    logger = logging.getLogger(name)          # use the name
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

