import os
import csv
import copy
import json
import argparse

import torch
import torch.distributed as dist
import numpy as np
from subject_aware_contrastive_learning.trainer.utils import (
    create_experiment, setup_logger, setup_distributed, save_results, resolve_seeds,broadcast_rank
)

from trainer.pretrainer import PreTrainer
from trainer.finetuner import Finetuner

BASE_OUTPUT = "./save"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to JSON config file.")
    parser.add_argument("--model_type", type=str, required=True, choices=["contrastive", "subject_specific", "subject_invariant"], help="model type for pretraining, contrastive or subject_specific")

    args = parser.parse_args()

    # Load config
    with open(args.config_path) as f:
        cfg = json.load(f)

     #Declare variables
    results = {'f1': [], 'acc': []}
    exp_name = cfg.get("exp_name", "exp")
    split_fold = cfg["split_path"]        
    folds = sorted(p for p in os.listdir(split_fold) if p.endswith(".csv"))

    #Check the cfg dataset
    if cfg['pretrain_args']['dataset_args']['train_dataset_args']['split'] is None:
        raise ValueError("Dataset split not specified in config. Please provide a split file for pretraining.")
    # Setup distributed training
    rank, world_size, device = setup_distributed()
    seeds = resolve_seeds(cfg)

     #Update the model type for pretraining
    cfg_run = copy.deepcopy(cfg)
    cfg_run['pretrain_args']["model_args"]["model_type"] = args.model_type
    if rank == 0: print(f"Runs: {len(seeds)}, seeds: {seeds}, model_type: {args.model_type}")

    if rank == 0: print(f"Runs: {len(seeds)}, seeds: {seeds}")

    for seed in seeds:
        # Create output dir on rank 0, then broadcast
        output_dir = (
            create_experiment(cfg_run["logging_args"]["base_output_dir"],
                              model_type = cfg_run['pretrain_args']['model_args']['model_type'] ,
                              exp_name=exp_name, mode="", dataset=cfg_run['pretrain_args']["dataset_args"]["train_dataset_args"]["data_name"])
            if rank == 0 else None
        )
        pretrain_dir = (
            create_experiment(output_dir, mode="pretrain")
            if rank == 0 else None
        )
        finetune_dir = (
            create_experiment(output_dir, mode="finetune")
            if rank == 0 else None
        )

        if world_size > 1:
            obj = [pretrain_dir, finetune_dir, output_dir]
            dist.broadcast_object_list(obj, src=0)
            pretrain_dir = obj[0]
            finetune_dir = obj[1]
            output_dir = obj[2]

        cfg_run["logging_args"]["output_dir"] = output_dir
        cfg_run["logging_args"]["pretrain_output_dir"] = pretrain_dir
        cfg_run["logging_args"]["finetune_output_dir"] = finetune_dir

        pretrain_logger = setup_logger(pretrain_dir)
        finetune_logger = setup_logger(finetune_dir)

        #Loop through all folds
        for run_idx, fold in enumerate(folds):
            cfg_run_fold = copy.deepcopy(cfg_run)
            #Update the split file for the current fold
            split_file = os.path.join(split_fold, fold)
            for split in ("train_dataset_args", "val_dataset_args", "test_dataset_args"):
                cfg_run_fold['finetune_args']["dataset_args"][split]["split_file"] = split_file #update finetune_aargs
            #Adding the split parameter for pretraining as well since we are doing LOSO pretraining
            cfg_run_fold['pretrain_args']['dataset_args']['train_dataset_args']["split_file"] = split_file #update pretrain_aargs
            
            #Pretraining and finetuning
            pretrain_out = PreTrainer(cfg_run_fold, logger=pretrain_logger, device=device, rank=rank, world_size=world_size, fold = run_idx).train()
            best_path = broadcast_rank(pretrain_out['best_path'] if rank == 0 else None, rank)
            cfg_run_fold['finetune_args']["model_args"]["model_path"] =  best_path#Update model path for finetuning
            save_results(pretrain_out, os.path.join(output_dir, "pretrain_results.csv"))

            finetune_out = Finetuner(cfg_run_fold, logger=finetune_logger, device=device, rank=rank, world_size=world_size, fold = run_idx).train()
            finetune_out = broadcast_rank(finetune_out if rank == 0 else None, rank)
            results['f1'].append(finetune_out['best_f1'])
            results['acc'].append(finetune_out['best_acc'])
            save_results(finetune_out, os.path.join(output_dir, "finetune_results.csv"))
            
        all_result = {'f1_score': np.mean(results['f1']), 'accuracy': np.mean(results['acc']),
                      "seeds": seeds, "Pretrain_path": pretrain_dir, "Finetune_path": finetune_dir}
        save_results(all_result, 
                        os.path.join(BASE_OUTPUT, f"results.csv"))


    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()