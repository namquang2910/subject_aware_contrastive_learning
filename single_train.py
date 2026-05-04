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

DATASET_DIC = {'WESADDataset':'/home/s223149341/SSL-invariance-Subject_Project_model/data/WESAD/wesad_10_05_no_standardize',
               'SWELLDataset':'/home/s223149341/SSL-invariance-Subject_Project_model/data/SWELL/SWELL_1280_320',
               'PsychioNet': '/home/s223149341/SSL-invariance-Subject_Project_model/data/PhysioNet2017/physionet2017_unlabelled_10_5.parquet',
               "STRESSIDDataset": '/home/s223149341/SSL-invariance-Subject_Project_model/data/STRESS_ID/STRESS_ID_1280_64',
               "VERBIODataset": '/home/s223149341/SSL-invariance-Subject_Project_model/data/verbio_1280_64'
               }

def resolve_args(args, cfg):
    # Resolve Pretrain Dataset
    if args.dataset is not None:
        if args.dataset not in DATASET_DIC:
            raise ValueError(f"Dataset {args.dataset} not found. ")
        if args.dataset == "PsychioNet_z":
            args.dataset = "PsychioNet"  # Use the same dataset class for both, but with different paths
        cfg['pretrain_args']['dataset_args']['train_dataset_args']['data_name'] = args.dataset
        cfg['pretrain_args']['dataset_args']['train_dataset_args']['dataset_path'] = DATASET_DIC[args.dataset]

    # Resolve Pretrain Model Type
    if args.model_type is not None:
        cfg['pretrain_args']["model_args"]["model_type"] = args.model_type

    # Resolve Dataset Split (Cross-dataset must use full dataset)
    split = cfg['pretrain_args']['dataset_args']['train_dataset_args']['split']
    
    if split is not None:
        raise ValueError("This is the cross-dataset setting. Please set 'split=None' for full dataset training.")
    
    if args.finetune_fraction is not None:
        print(f"Setting finetune dataset fraction to {args.finetune_fraction}")
        cfg['finetune_args']["dataset_args"]['train_dataset_args']['sub_sample_frac'] = args.finetune_fraction
        
    return cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--resume_finetune", type=int, default=-1 , help="continue to finetune from a previous pretrain run")
    parser.add_argument("--model_type", type=str, required=True, choices=["contrastive", "subject_specific", "subject_invariant", "moe_dual_branch", "byol", "simsiam"], help="model type for pretraining, contrastive or subject_specific")
    parser.add_argument("--model_path", type=str, default=None , help="Path to model for finetune")
    parser.add_argument("--dataset", type=str, default=None , choices=["WESADDataset", "SWELLDataset", "PsychioNet", "PsychioNet_z", "STRESSIDDataset", "VERBIODataset"], help="Pretrain dataset")
    parser.add_argument("--finetune_fraction", type=float, default=None , help="continue to finetune from a previous pretrain run")
    args = parser.parse_args()

    with open(args.config_path) as f:
        cfg = json.load(f)
    allow_exit = False
    all_seed_results = []

    #Resolve the Arguments and config file
    cfg = resolve_args(args, cfg)
    
    #Setup the DDP
    rank, world_size, device = setup_distributed()
    seeds = resolve_seeds(cfg)
    cfg_run = copy.deepcopy(cfg)

    #Update the model type for pretraining
    if rank == 0: print(f"Runs: {len(seeds)}, seeds: {seeds}, model_type: {args.model_type}")
    try:
        for seed in seeds:
            seed_results = {
                "seed": seed,
                "f1": [],
                "acc": [],
            }
            if args.resume_finetune >= 0:
                allow_exit = True
                print(f"Resuming finetune from pretrain run with seed {seed} and fold {args.resume_finetune}")
            pretrain_output_dir = (
                create_experiment(cfg_run["logging_args"]["base_output_dir"],
                                  model_type=cfg_run['pretrain_args']['model_args']['model_type'],
                                  exp_name=cfg.get("exp_name", "exp"), mode="pretrain", 
                                  seed= seed,
                                  dataset=cfg_run['pretrain_args']["dataset_args"]['train_dataset_args']['data_name'], allow_exist=allow_exit)
                if rank == 0 else None
            )
            finetune_output_dir = (
                create_experiment(cfg_run["logging_args"]["base_output_dir"],
                                  model_type=cfg_run['pretrain_args']['model_args']['model_type'],
                                  exp_name=cfg.get("exp_name", "exp"), mode="finetune", dataset=cfg_run['pretrain_args']["dataset_args"]['train_dataset_args']['data_name'], 
                                  seed = seed,
                                  finetune_dataset=cfg_run['finetune_args']["dataset_args"]['train_dataset_args']['data_name'],
                                  allow_exist=allow_exit)
                if rank == 0 else None
            )
            if world_size > 1:
                obj = [pretrain_output_dir, finetune_output_dir]
                dist.broadcast_object_list(obj, src=0)
                pretrain_output_dir = obj[0]
                finetune_output_dir = obj[1]

            cfg_run["logging_args"]["pretrain_output_dir"] = pretrain_output_dir
            cfg_run["logging_args"]["finetune_output_dir"] = finetune_output_dir
            
            pretrain_logger = setup_logger(pretrain_output_dir, name="pretrain")
            finetune_logger = setup_logger(finetune_output_dir, name="finetune")

            if args.resume_finetune >= 0:
                finetune_logger.info(f"Resuming finetune from pretrain run with seed {seed} and fold {args.resume_finetune}")
                if args.model_path is not None:             
                    cfg_run['finetune_args']["model_args"]["model_path"]  = args.model_path
                else:   
                    cfg_run['finetune_args']["model_args"]["model_path"] = os.path.join(pretrain_output_dir, "encoder_best_.pt")
            else:
                pretrain_logger.info(f"Starting new run with seed {seed}")
                pretrain_out = PreTrainer(cfg_run, logger=pretrain_logger, device=device, rank=rank, world_size=world_size, fold="", seed = seed).train()
                best_path = broadcast_rank(pretrain_out['best_path'] if rank == 0 else None, rank)
                cfg_run['finetune_args']["model_args"]["model_path"] = best_path
                save_results(pretrain_out, os.path.join(pretrain_output_dir, "results.csv")) if rank == 0 else None
                
            
            split_fold = cfg["split_path"]
            folds = sorted(p for p in os.listdir(split_fold) if p.endswith(".csv"))
            for run_id, _ in enumerate(folds):
                finetune_cfg = copy.deepcopy(cfg_run)
                if args.resume_finetune >= 0 and run_id < args.resume_finetune:
                    finetune_logger.info(f"Skipping fold {run_id} as per resume_finetune={args.resume_finetune}")
                    continue

                #Update the split file for finetuning
                split_file = os.path.join(split_fold, folds[run_id])
                for split in ["train_dataset_args", "val_dataset_args", "test_dataset_args"]:
                    print(f"Updating split file for fold {run_id}: {split_file}")
                    finetune_cfg['finetune_args']["dataset_args"][split]["split_file"] = split_file

                finetuner = Finetuner(finetune_cfg, logger=finetune_logger, device=device, rank=rank, world_size=world_size, fold=run_id, seed = seed)
                finetune_out = finetuner.train()    
                finetune_out['training_mode'] = cfg_run['finetune_args']["model_args"].get('training_mode', None)
                finetune_out = broadcast_rank(finetune_out if rank == 0 else None, rank)
                seed_results["f1"].append(finetune_out['best_f1'])
                seed_results["acc"].append(finetune_out['best_acc'])
                if rank == 0: print (f"Fold {run_id} - F1: {finetune_out['best_f1']:.4f}, Acc: {finetune_out['best_acc']:.4f} Saving finetune results for fold {run_id} in to {finetune_output_dir}")
                save_results(finetune_out, os.path.join(finetune_output_dir, "results.csv")) if rank == 0 else None

          # ---------------- PER-SEED AVG ---------------- #

            seed_results = {'f1_score': np.mean(seed_results['f1']), 
                          'accuracy': np.mean(seed_results['acc']), 
                          "seed": seed, 
                          "fraction": cfg_run['finetune_args']["dataset_args"]['train_dataset_args'].get('sub_sample_frac', None),
                          "Pretrain_output": pretrain_output_dir, 
                          "Finetune_output": finetune_output_dir,
                          "training_mode": cfg_run['finetune_args']["model_args"].get('training_mode', None)}
            path = os.path.join(BASE_OUTPUT, "results.json")
            if rank == 0:             
                save_results(
                seed_results,
                path
            )
            
            print(f"Saving seedresults for seed {seed} in to {path}")
            all_seed_results.append(seed_results)
        # ---------------- CROSS-SEED AVG ---------------- #
        if len(seeds) > 1:
            final_results = {
                "mean_f1_across_seeds":
                    np.mean([s["f1_score"] for s in all_seed_results]),
                "std_f1_across_seeds":
                    np.std([s["f1_score"] for s in all_seed_results]),
                "mean_acc_across_seeds":
                    np.mean([s["accuracy"] for s in all_seed_results]),
                "std_acc_across_seeds":
                    np.std([s["accuracy"] for s in all_seed_results]),

                "seeds": seeds
            }

            if rank == 0:
                save_results(
                    final_results,
                    os.path.join(BASE_OUTPUT, "summary_results.csv")
                )
    except Exception as e:
        if rank == 0 and pretrain_output_dir is not None and os.path.exists(pretrain_output_dir):
            print(f"Error occurred: {e}. Removing output directory: {pretrain_output_dir}")
        raise  # re-raise so DDP workers still get the traceback

    finally:
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()