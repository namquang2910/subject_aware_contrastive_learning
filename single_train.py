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
    parser.add_argument("--resume_finetune", type=int, default=-1 , help="continue to finetune from a previous pretrain run")
    parser.add_argument("--model_type", type=str, required=True, choices=["contrastive", "subject_specific", "subject_invariant", "moe_dual_branch"], help="model type for pretraining, contrastive or subject_specific")
    parser.add_argument("--model_path", type=str, default=None , help="continue to finetune from a previous pretrain run")
    args = parser.parse_args()

    with open(args.config_path) as f:
        cfg = json.load(f)
    allow_exit = False
    results = {'f1': [], 'acc': []}
    exp_name = cfg.get("exp_name", "exp")

    if cfg['pretrain_args']['dataset_args']['train_dataset_args']['split'] is not None:
            raise ValueError("This is the cross dataset setting. Please set 'split' to None for full dataset training.")

    rank, world_size, device = setup_distributed()
    seeds = resolve_seeds(cfg)
    cfg_run = copy.deepcopy(cfg)
    output_dir = None  # track for cleanup

    #Update the model type for pretraining
    cfg_run['pretrain_args']["model_args"]["model_type"] = args.model_type
    if rank == 0: print(f"Runs: {len(seeds)}, seeds: {seeds}, model_type: {args.model_type}")
    try:
        for seed in seeds:
            if args.resume_finetune >= 0:
                allow_exit = True
                print(f"Resuming finetune from pretrain run with seed {seed} and fold {args.resume_finetune}")
            output_dir = (
                create_experiment(cfg_run["logging_args"]["base_output_dir"],
                                  model_type=cfg_run['pretrain_args']['model_args']['model_type'],
                                  exp_name=exp_name, mode="", dataset=cfg_run['pretrain_args']["dataset_args"]['train_dataset_args']['data_name'], allow_exist=allow_exit)
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
            if args.resume_finetune >= 0:
                logger.info(f"Resuming finetune from pretrain run with seed {seed} and fold {args.resume_finetune}")                
                if args.model_path is not None:
                    cfg_run['finetune_args']["model_args"]["model_path"] = args.model_path
                else:
                    cfg_run['finetune_args']["model_args"]["model_path"] = os.path.join(output_dir, "encoder_best_.pt")
            else:
                logger.info(f"Starting new run with seed {seed}")
                pretrain_out = PreTrainer(cfg_run, logger=logger, device=device, rank=rank, world_size=world_size, fold="", seed = seed).train()
                best_path = broadcast_rank(pretrain_out['best_path'] if rank == 0 else None, rank)
                cfg_run['finetune_args']["model_args"]["model_path"] = best_path
                save_results(pretrain_out, os.path.join(output_dir, "results.csv")) if rank == 0 else None

            if world_size > 1:
                dist.barrier()  # ← ADD THIS: ensure both ranks sync before finetuner init
            split_fold = cfg["split_path"]
            folds = sorted(p for p in os.listdir(split_fold) if p.endswith(".csv"))

            for run_id, _ in enumerate(folds):
                finetune_cfg = copy.deepcopy(cfg_run)
                if args.resume_finetune >= 0 and run_id < args.resume_finetune:
                    logger.info(f"Skipping fold {run_id} as per resume_finetune={args.resume_finetune}")
                    continue

                #Update the split file for finetuning
                split_file = os.path.join(split_fold, folds[run_id])
                print(f"Updating split file for fold {run_id}: {split_file}")
                for split in ["train_dataset_args", "val_dataset_args", "test_dataset_args"]:
                    finetune_cfg['finetune_args']["dataset_args"][split]["split_file"] = split_file

                finetune_out = Finetuner(finetune_cfg, logger=logger, device=device, rank=0, world_size=1, fold=run_id).train()
                #finetune_out = broadcast_rank(finetune_out if rank == 0 else None, rank)
                results['f1'].append(finetune_out['best_f1'])
                results['acc'].append(finetune_out['best_acc'])
                if rank == 0: print (f"Fold {run_id} - F1: {finetune_out['best_f1']:.4f}, Acc: {finetune_out['best_acc']:.4f} Saving finetune results for fold {run_id} in to {output_dir}")
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
            print(f"Error occurred: {e}. Output directory: {output_dir}")
            user_input = input("Remove output directory? (y/n): ").strip().lower()
            if user_input == 'y':
                shutil.rmtree(output_dir)
                print(f"Removed: {output_dir}")
            else:
                print(f"Keeping: {output_dir}")
        
        if world_size > 1:
            dist.barrier()  # let non-rank-0 workers exit cleanly
        raise
    finally:
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()