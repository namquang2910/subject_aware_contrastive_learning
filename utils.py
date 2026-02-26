"""
Utility functions to support model training
"""
import json
import time
import random
import csv
import os 
from models import contrastive_model, subject_invariant_model, subject_specific_model
import torch
import numpy as np
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from models.utils import get_base_encoder
import random
from torch.utils.data import Sampler
from datasets.wesad_dataset import WESADDataset
from datasets.psy_dataset import PsyDataset
from datasets.swell_dataset import SWELLDataset
from collections import defaultdict
import torch.distributed as dist

def broadcast_rank(obj, rank):
    """
    Broadcast a Python object from rank 0 to all ranks.

    Args:
        obj: Object to broadcast (only required on rank 0, others can pass None)
        rank: Current process rank

    Returns:
        The broadcasted object (same on all ranks)
    """
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
        
def create_experiment(base_dir, model_type = "", exp_name="", dataset="", mode=""):
    tag = f"{model_type}_{dataset}" if model_type != "" or dataset != "" else ""
    out = os.path.join(base_dir, exp_name,tag, mode)
    
    if os.path.exists(out):
        print(f"Warning: path {out} already exists. Overwriting.")
        exit(1)
    else:
        os.makedirs(out, exist_ok=True)
    return out

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


