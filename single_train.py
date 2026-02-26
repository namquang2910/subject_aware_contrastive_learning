import os
import csv
import copy
import json
import argparse

import torch
import shutil
import torch.distributed as dist
import numpy as np
from utils import (
    create_experiment, setup_logger, setup_distributed, save_results, resolve_seeds,broadcast_rank
)

from trainer.pretrainer import PreTrainer
from trainer.finetuner import Finetuner

BASE_OUTPUT = "./save"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()

    with open(args.config_path) as f:
        cfg = json.load(f)

    results = {'f1': [], 'acc': []}
    exp_name = cfg.get("exp_name", "exp")

    rank, world_size, device = setup_distributed()
    seeds = resolve_seeds(cfg)
    cfg_run = copy.deepcopy(cfg)
    output_dir = None  # track for cleanup

    try:
        for seed in seeds:
            output_dir = (
                create_experiment(cfg["logging_args"]["base_output_dir"],
                                  model_type=cfg['pretrain_args']['model_args']['model_type'],
                                  exp_name=exp_name, mode="", dataset=cfg["dataset_args"]["data_name"])
                if rank == 0 else None
            )

            if world_size > 1:
                obj = [output_dir]
                dist.broadcast_object_list(obj, src=0)
                output_dir = obj[0]

            cfg_run["logging_args"]["output_dir"] = output_dir
            cfg_run["logging_args"]["pretrain_output_dir"] = output_dir
            cfg_run["logging_args"]["finetune_output_dir"] = output_dir

            logger = setup_logger(output_dir)

            pretrain_out = PreTrainer(cfg_run, logger=logger, device=device, rank=rank, world_size=world_size, fold="").train()
            best_path = broadcast_rank(pretrain_out['best_path'] if rank == 0 else None, rank)
            cfg_run['finetune_args']["model_args"]["model_path"] = best_path
            save_results(pretrain_out, os.path.join(output_dir, "results.csv")) if rank == 0 else None

            split_fold = cfg["split_path"]
            folds = sorted(p for p in os.listdir(split_fold) if p.endswith(".csv"))

            for run_id, _ in enumerate(folds):
                split_file = os.path.join(split_fold, folds[run_id])
                for split, split_type in zip(
                    ("train_dataset_args", "val_dataset_args", "test_dataset_args"),
                    ("train", "val", "test")
                ):
                    cfg_run["dataset_args"][split]["split"] = split_type
                    cfg_run["dataset_args"][split]["split_file"] = split_file

                finetune_out = Finetuner(cfg_run, logger=logger, device=device, rank=rank, world_size=world_size, fold=run_id).train()
                finetune_out = broadcast_rank(finetune_out if rank == 0 else None, rank)
                results['f1'].append(finetune_out['best_f1'])
                results['acc'].append(finetune_out['best_acc'])
                save_results(finetune_out, os.path.join(output_dir, "results.csv")) if rank == 0 else None

        all_result = {
            'f1_score': np.mean(results['f1']),
            'accuracy': np.mean(results['acc']),
            "seeds": seeds,
            "Output_path": output_dir
        }
        if rank == 0:
            save_results(all_result, os.path.join(BASE_OUTPUT, "results.csv"))

    except Exception as e:
        if rank == 0 and output_dir is not None and os.path.exists(output_dir):
            print(f"Error occurred: {e}. Removing output directory: {output_dir}")
            shutil.rmtree(output_dir)
        raise  # re-raise so DDP workers still get the traceback

    finally:
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()