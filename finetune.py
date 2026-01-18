import csv
import os
import json
import time
import copy
import argparse
import inspect
import math
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from loss.cl_loss import NCELoss
import torch.nn as nn
import numpy as np
import pandas as pd
# project imports
from models.utils import get_base_encoder
from utils import set_seed, create_timestamped_subdir, save_config_file, setup_logger, EarlyStopping, compute_metrics
from models import finetune_builder
from loss import cl_loss
from datasets import wesad_dataset
from torch.optim.lr_scheduler import CosineAnnealingLR


class Trainer:
    def __init__(self, cfg, logger, run_idx):
        self.cfg = cfg                      # <-- FIXED
        self.logger = logger
        self.run_idx = run_idx

        # --- Seed & device ---
        set_seed(cfg["seed"])
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.output_dir = cfg["output_dir"]

        # --- Loss ---
        self.loss_fn = nn.BCEWithLogitsLoss()

        # --- Model ---
        enc_name = cfg["model_args"]["base_encoder"]
        enc_args = cfg["model_args"]["base_encoder_args"]
        encoder = get_base_encoder(enc_name, enc_args)
        print(encoder)
        self.model = finetune_builder.EncoderClassifierModel(
            base_encoder=encoder,
            num_class=cfg["model_args"]["num_class"],
            model_path=cfg["model_args"]["model_path"],
            device = self.device,
            freeze_encoder=cfg["model_args"].get("freeze_encoder", False)
        )
        self.model.set_loss_fn(self.loss_fn)
        self.model.check_grad_status()
        self.model.to(self.device)
        encoder_param, classifier_params = self.model.get_parameters() 
        self.logger.info(f"Model.forward signature: {inspect.signature(self.model.forward)}")

        # --- Optimizer & Scheduler ---
        opt_cfg = cfg["optim_args"]
        min_lr_value = opt_cfg["min_lr"]
        self.epochs = int(opt_cfg["epochs"])
        self.optimizer = optim.AdamW(
            [
                {"params": encoder_param, "lr": opt_cfg["lr"]/10},
                {"params": classifier_params, "lr": opt_cfg["lr"]},
            ],
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg.get("weight_decay", 0.0),
            betas=(opt_cfg.get("adam_beta1", 0.9), opt_cfg.get("adam_beta2", 0.999)),
            eps=opt_cfg.get("adam_epsilon", 1e-8),
        )

        self.use_lr_scheduler = opt_cfg.get("use_lr_scheduler", False)
        if self.use_lr_scheduler:
            self.warm_up_epochs = opt_cfg.get("warm_up", 0)
            if self.warm_up_epochs is None:
                self.warm_up_epochs = 0
                self.logger.warning("'warm_up' not specified in optimizer config; using 0")

            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, self.epochs - self.warm_up_epochs),
                eta_min=min_lr_value,
            )
        else:
            self.scheduler = None

        # --- Early Stopping ---
        self.early_stopper = EarlyStopping(
            min_delta=opt_cfg.get("min_delta", 1e-4),
            patience=opt_cfg.get("patience", 10),
            enabled=opt_cfg.get("apply_early_stopping", True),
            is_higher=opt_cfg.get("early_stopping_is_higher", False)
        )

        # --- Dataloaders: LOSO vs Train/Test ---
        ds_cfg = cfg["dataset_args"]
        eval_type = ds_cfg["evaluation_type"]

        if eval_type == "loso":
            # config expected:
            # "dataset_args": {
            #     "evaluation_type": "loso",
            #     "train_dataset": {...},
            #     "val_dataset": {...},
            #     "test_dataset": {...}
            # }
            train_args = ds_cfg["train_dataset_args"]
            val_args = ds_cfg["val_dataset_args"]
            test_args = ds_cfg["test_dataset_args"]

            self.train_ds = wesad_dataset.WESADDataset(**train_args)
            self.val_ds = wesad_dataset.WESADDataset(**val_args)
            self.test_ds = wesad_dataset.WESADDataset(**test_args)

            self.train_loader = self.create_data_loader(self.train_ds, shuffle=True, drop_last=True)
            self.val_loader = self.create_data_loader(self.val_ds, shuffle=False)
            self.test_loader = self.create_data_loader(self.test_ds, shuffle=False)
            self.logger.info(
                f"[LOSO] Train size: {len(self.train_ds)}, "
                f"Val size: {len(self.val_ds)}, "
                f"Test size: {len(self.test_ds)}"
            )

        elif eval_type == "train_test":
            # config expected:
            # "dataset_args": {
            #     "evaluation_type": "train_test",
            #     "dataset": {...},       # args for FinetunedDataset (full)
            #     "train_ratio": 0.7,
            #     "train_subsample_frac": 0.01 (optional)
            # }
            data_args = ds_cfg
            full_dataset = finetune_dataset.FinetunedDataset(**data_args["dataset"])

            total_len = len(full_dataset)
            train_ratio = data_args["train_ratio"]
            train_len = int(total_len * train_ratio)
            val_len = int((total_len - train_len) * 0.5)
            test_len = total_len - train_len - val_len

            train_indices = range(0, train_len)
            val_indices = range(train_len, train_len + val_len)
            test_indices = range(train_len + val_len, total_len)

            train_dataset = Subset(full_dataset, train_indices)
            val_dataset = Subset(full_dataset, val_indices)
            test_dataset = Subset(full_dataset, test_indices)

            # Optional: subsample only train set
            subsample_frac = data_args.get("train_subsample_frac", None)
            if subsample_frac is not None:
                train_dataset = self.subsample_subset(train_dataset, subsample_frac)
                self.logger.info(f"Subsampled train to fraction={subsample_frac}")

            self.train_loader = self.create_data_loader(train_dataset, shuffle=True)
            self.val_loader = self.create_data_loader(val_dataset, shuffle=False)
            self.test_loader = self.create_data_loader(test_dataset, shuffle=False)

            self.logger.info(
                f"[Train/Test] Total: {total_len}, "
                f"Train size: {len(train_dataset)}, "
                f"Val size: {len(val_dataset)}, "
                f"Test size: {len(test_dataset)}"
            )

        else:
            raise ValueError(f"Unknown evaluation_type={eval_type}")

        self.logger.info(f"Train batches per epoch: {len(self.train_loader)}")
        # --- Save config ---
        save_config_file(cfg, self.output_dir)

    # ---------- Helpers ----------

    def subsample_subset(self, subset, fraction):
        """Return a new Subset containing a random fraction of an existing Subset."""
        original_indices = list(subset.indices)
        n = len(original_indices)
        k = max(1, int(n * fraction))  # keep at least 1 sample

        keep = np.random.choice(original_indices, size=k, replace=False)
        keep = sorted(keep)
        return Subset(subset.dataset, keep)

    def create_data_loader(self, dataset, shuffle, drop_last=False):
        return DataLoader(
            dataset,
            batch_size=self.cfg["optim_args"]["batch_size"],
            shuffle=shuffle,
            num_workers=self.cfg["optim_args"]["num_workers"],
            pin_memory=True,
            drop_last=drop_last,
        )

    # ---------- Training / Validation ----------

    def train_one_epoch(self, epoch):
        start_time = time.time()
        self.model.train()
        losses = 0.0

        for batch in self.train_loader:
            x = batch["x"].to(self.device, non_blocking=True).float()
            y = batch["y"].to(self.device, non_blocking=True).float()  # BCEWithLogitsLoss expects float
            self.model.zero_grad()
            # assuming model(x, y) returns loss
            loss = self.model(x, y)
            loss.backward()
            self.optimizer.step()

            losses += loss.item()

        if self.scheduler is not None and epoch >= self.warm_up_epochs:
            self.scheduler.step()

        avg_loss = losses / max(1, len(self.train_loader))
        elapsed = time.time() - start_time

        if (epoch % self.cfg["logging_args"]["print_freq"]) == 0:
            last_lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info(
                f"[Epoch {epoch:03d}] loss={avg_loss:.6f} lr={last_lr:.6f} time={elapsed:.2f}s"
            )
        return avg_loss

    def train(self):
        best_path = None
        best_result = None
        for epoch in range(self.epochs):
            print(f"Starting epoch {epoch}")
            loss = self.train_one_epoch(epoch)
            val_loss, result = self.validate(self.val_loader)
            should_stop, improved = self.early_stopper.step(result["f1"])

            if improved:
                best_result = result
                self.logger.info(f"---------- New best model found at epoch {epoch} with F1={result['f1']:.6f}------------")
                best_path = os.path.join(self.output_dir, f"encoder_best.pt")
                torch.save(self.model.state_dict(), best_path)
                print("--------- Model is the best right now ---------")

            if should_stop:
                self.logger.info(
                    f"Early stopping at epoch {epoch}. Best loss: {self.early_stopper.best:.6f}"
                )
                break

        if best_path is not None:
            state = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(state)

        self.logger.info("Training complete.")
        self.logger.info(f"Best result: {best_result}")
        return best_result
    
    def validate(self, dataloader, return_cm=False):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                x = batch['x'].to(self.device).float()
                y = batch['y'].to(self.device).float()

                loss, preds = self.model(x, y, return_preds=True)
                total_loss += loss.item()

                probs = torch.sigmoid(preds)
                y_hat = (probs >= 0.5).long().view(-1)

                all_preds.append(y_hat.cpu())
                all_labels.append(y.view(-1).long().cpu())

        avg_loss = total_loss / max(1, len(dataloader))

        y_true = torch.cat(all_labels)
        y_pred = torch.cat(all_preds)
        result = compute_metrics(y_true, y_pred)
        log_data = f"Validation — loss: {avg_loss:.4f} - Accuracy: {result['acc']} - F1score: {result['f1']} - Recall: {result['recall']} - Precision {result['precision']}"
        if return_cm:
            log_data += f" - Confusion Matrix: {result['conf_mat']}"
        self.logger.info(log_data)
        print(log_data)

        return avg_loss, result

  
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to JSON config file.")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        cfg = json.load(f)

    # --- normalize seeds ---
    if "seeds" in cfg and isinstance(cfg["seeds"], list) and len(cfg["seeds"]) > 0:
        seeds = cfg["seeds"]
    elif "seed" in cfg and isinstance(cfg["seed"], int):
        seeds = [cfg["seed"]]
    else:
        seeds = [42]

    ds_cfg = cfg["dataset_args"]
    eval_type = ds_cfg["evaluation_type"]

    # --- Output dir ---
    log_cfg = cfg["logging_args"]
    base_out = log_cfg["base_output_dir"]
    exp_name = cfg.get("exp_name", "exp")
    out_dir = create_timestamped_subdir(base_out, exp_name=exp_name)

    num_runs = len(seeds)
    print(f"[Runs] num_runs={num_runs}, seeds={seeds}")

    f1_ls, acc_ls = [], []

    # ========== LOSO MODE ==========
    if eval_type == "loso":
        split_fold = cfg["split_path"]   # folder containing per-fold CSVs
        all_fold = [p for p in os.listdir(split_fold) if p.endswith(".csv")]
        all_fold.sort()

        for seed in seeds:
            for run_idx, fold in enumerate(all_fold):
                split_file = os.path.join(split_fold, fold)

                # ----- Update configuration for this run & fold -----
                cfg_run = copy.deepcopy(cfg)
                cfg_run["seed"] = int(seed)
                cfg_run["run_num"] = run_idx
                # point train/val/test split to this file
                for data in ["train_dataset_args", "val_dataset_args", "test_dataset_args"]:
                    cfg_run["dataset_args"][data]["split_file"] = split_file

                cfg_run["output_dir"] = out_dir

                # logger per run
                logger = setup_logger(out_dir)
                logger.info(f"=== Run {run_idx+1}/{num_runs} | seed={seed} | fold={fold} ===")
                logger.info(f"Loading split file {split_file}")

                trainer = Trainer(cfg_run, logger, run_idx)
                result = trainer.train()
                #_, result = trainer.validate(trainer.test_loader, return_cm=True)
                f1_ls.append(result["f1"])
                acc_ls.append(result["acc"])

                row_df = pd.DataFrame({
                    "f1_scores": [result["f1"]],
                    "accuracy": [result["acc"]],
                    "split_fold": [fold]
                })
                csv_path = os.path.join(out_dir, "loso_results.csv")
                file_exists = os.path.isfile(csv_path)
                row_df.to_csv(
                    csv_path,
                    mode="a",           # append to file
                    header=not file_exists,  # write header only if file does not exist
                    index=False
                )
        print(f"LOSO Mean F1 score {np.mean(f1_ls)} - Accuracy mean {np.mean(acc_ls)}")
        print(f"LOSO results saved to {out_dir}")
        print("--------- Training is done ---------")
        save_file(f1_ls, acc_ls, cfg)

    # ========== TRAIN / TEST MODE ==========
    elif eval_type == "train_test":
        # no LOSO folds; just run seeds
        for run_idx, seed in enumerate(seeds):
            cfg_run = copy.deepcopy(cfg)
            cfg_run["seed"] = int(seed)
            cfg_run["run_num"] = run_idx
            cfg_run["output_dir"] = out_dir

            logger = setup_logger(out_dir)
            logger.info(f"=== Run {run_idx+1}/{num_runs} | seed={seed} ===")

            trainer = Trainer(cfg_run, logger, run_idx)
            trainer.train()

            _, result = trainer.validate(trainer.test_loader, return_cm=True)
            f1_ls.append(result["f1"])
            acc_ls.append(result["acc"])

    else:
        raise ValueError(f"Unknown evaluation_type={eval_type}")

    print(f"Mean F1 score {np.mean(f1_ls)} - Accuracy mean {np.mean(acc_ls)}")

def save_file(f1_ls, acc_ls,cfg):
    header = [
        "F1_score",
        "Accuracy",
        "Pretraining_model_path",
        "Finetuned_model_path"
        ]
    row = [
        np.mean(f1_ls),
        np.mean(acc_ls),
        cfg["model_args"]["model_path"],
        cfg["output_dir"]
        ]

    file_exists = os.path.isfile(cfg["logging_args"]["results_file"])

    with open(cfg["logging_args"]["results_file"], 'a', newline='') as fd:
        writer = csv.writer(fd)
        # write header only once
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)

if __name__ == "__main__":
    main()
