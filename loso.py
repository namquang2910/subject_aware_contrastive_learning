import os
import csv
import copy
import json
import time
import inspect
import argparse

from subject_aware_contrastive_learning.finetune import Trainer, save_results
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

from loss.cl_loss import NCELoss
from utils import (
    LossMeter, EarlyStopping,
    create_experiment, create_model,
    get_dataset, save_config_file,
    set_seed, setup_logger, DummyLogger
)

from trainer.pretrainer import PreTrainer
from trainer.finetuner import Finetuner

BASE_OUTPUT = ".save"
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to JSON config file.")
    args = parser.parse_args()

    #Declare variables
    results = {'f1': [], 'acc': []}
    exp_name = cfg.get("exp_name", "exp")
    split_fold = cfg["split_path"]        
    folds = sorted(p for p in os.listdir(split_fold) if p.endswith(".csv"))

    # Load config
    with open(args.config_path) as f:
        cfg = json.load(f)

    # Setup distributed training
    rank, world_size, device = setup_distributed()
    seeds = resolve_seeds(cfg)

    
    if rank == 0: print(f"Runs: {len(seeds)}, seeds: {seeds}")

    for seed in seeds:
        for run_idx, fold in enumerate(folds):
            cfg_run = copy.deepcopy(cfg) 
            
            split_file = os.path.join(split_fold, fold)
            cfg_run.update({"seed": int(seed), "run_num": run_idx, "output_dir": out_dir})
            for split in ("train_dataset_args", "val_dataset_args", "test_dataset_args"):
                cfg_run["dataset_args"][split]["split_file"] = split_file
            
            

            # Create output dir on rank 0, then broadcast
            pretrain_dir = (
                create_experiment(cfg["logging_args"]["base_output_dir"], exp_name=exp_name, mode="pretrain", dataset=cfg_run["dataset_args"]["data_name"])
                if rank == 0 else None
            )
            finetune_dir = (
                create_experiment(cfg["logging_args"]["base_output_dir"], exp_name=exp_name, mode="finetune", dataset=cfg_run["dataset_args"]["data_name"])
                if rank == 0 else None
            )

            if world_size > 1:
                obj = [pretrain_dir, finetune_dir]
                dist.broadcast_object_list(obj, src=0)
                pretrain_dir = obj[0]
                finetune_dir = obj[1]

            cfg_run["pretrain_output_dir"] = pretrain_dir
            cfg_run["finetune_output_dir"] = finetune_dir

            pretrain_logger = setup_logger(pretrain_dir)
            finetune_logger = setup_logger(finetune_dir)

            pretrain_out = PreTrainer(cfg_run, logger=pretrain_logger, device=device, rank=rank, world_size=world_size).train()
            cfg_run["model_args"]["model_path"] = pretrain_out['best_path'] #Update model path for finetuning
            save_results(pretrain_out, os.path.join(pretrain_dir, "pretrain_results.csv"))

            finetune_out = Finetuner(cfg_run, logger=finetune_logger, device=device, rank=rank, world_size=world_size).train()
            save_results(finetune_out, os.path.join(finetune_dir, "finetune_results.csv"))
            results['f1'].append(finetune_out['best_f1'])
            results['acc'].append(finetune_out['best_acc'])

        all_result = {'f1_score': np.mean(results['f1']), 'accuracy': np.mean(results['acc']),
                      "seeds": seeds, "Pretrain_path": pretrain_dir, "Finetune_path": finetune_dir}
        
        save_results(all_result, 
                     os.path.join(BASE_OUTPUT, f"results.csv"))


    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()