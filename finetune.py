import os
import csv
import copy
import json
import time
import inspect
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.utils import get_base_encoder
from models import finetune_builder
from datasets import wesad_dataset
from utils import (
    EarlyStopping,
    compute_metrics,
    create_timestamped_subdir,
    save_config_file,
    set_seed,
    setup_logger,
)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, cfg: dict, logger, run_idx: int):
        self.cfg = cfg
        self.logger = logger
        self.run_idx = run_idx

        set_seed(cfg["seed"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = cfg["output_dir"]

        self._build_model()
        self._build_optimizer()
        self._build_early_stopper()
        self._build_dataloaders()

        save_config_file(cfg, self.output_dir)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_model(self):
        m = self.cfg["model_args"]
        encoder = get_base_encoder(m["base_encoder"], m["base_encoder_args"])
        self.model = finetune_builder.EncoderClassifierModel(
            base_encoder=encoder,
            num_class=m["num_class"],
            model_path=m["model_path"],
            device=self.device,
            freeze_encoder=m.get("freeze_encoder", False),
        )
        self.model.set_loss_fn(nn.BCEWithLogitsLoss())
        self.model.check_grad_status()
        self.model.to(self.device)
        self.logger.info(f"Model.forward signature: {inspect.signature(self.model.forward)}")

    def _build_optimizer(self):
        opt = self.cfg["optim_args"]
        self.epochs = int(opt["epochs"])

        encoder_params, classifier_params = self.model.get_parameters()
        self.optimizer = optim.AdamW(
            [
                {"params": encoder_params, "lr": opt["lr"] / 10},
                {"params": classifier_params, "lr": opt["lr"]},
            ],
            lr=opt["lr"],
            weight_decay=opt.get("weight_decay", 0.0),
            betas=(opt.get("adam_beta1", 0.9), opt.get("adam_beta2", 0.999)),
            eps=opt.get("adam_epsilon", 1e-8),
        )

        self.warm_up_epochs = 0
        if opt.get("use_lr_scheduler", False):
            self.warm_up_epochs = opt.get("warm_up", 0) or 0
            if not self.warm_up_epochs:
                self.logger.warning("'warm_up' not specified; using 0.")
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, self.epochs - self.warm_up_epochs),
                eta_min=opt["min_lr"],
            )
        else:
            self.scheduler = None

    def _build_early_stopper(self):
        opt = self.cfg["optim_args"]
        self.early_stopper = EarlyStopping(
            min_delta=opt.get("min_delta", 1e-4),
            patience=opt.get("patience", 10),
            enabled=opt.get("apply_early_stopping", True),
            is_higher=opt.get("early_stopping_is_higher", False),
        )

    def _build_dataloaders(self):
        ds_cfg = self.cfg["dataset_args"]
        eval_type = ds_cfg["evaluation_type"]

        if eval_type == "loso":
            self.train_loader = self._make_loader(
                wesad_dataset.WESADDataset(**ds_cfg["train_dataset_args"]), shuffle=True, drop_last=True
            )
            self.val_loader = self._make_loader(
                wesad_dataset.WESADDataset(**ds_cfg["val_dataset_args"]), shuffle=False
            )
            self.test_loader = self._make_loader(
                wesad_dataset.WESADDataset(**ds_cfg["test_dataset_args"]), shuffle=False
            )
            self.logger.info(
                f"[LOSO] Train={len(self.train_loader.dataset)}, "
                f"Val={len(self.val_loader.dataset)}, "
                f"Test={len(self.test_loader.dataset)}"
            )

        elif eval_type == "train_test":
            full_ds = finetune_dataset.FinetunedDataset(**ds_cfg["dataset"])
            n = len(full_ds)
            train_len = int(n * ds_cfg["train_ratio"])
            val_len = int((n - train_len) * 0.5)

            train_ds = Subset(full_ds, range(0, train_len))
            val_ds = Subset(full_ds, range(train_len, train_len + val_len))
            test_ds = Subset(full_ds, range(train_len + val_len, n))

            if frac := ds_cfg.get("train_subsample_frac"):
                train_ds = self._subsample(train_ds, frac)
                self.logger.info(f"Subsampled train to fraction={frac}")

            self.train_loader = self._make_loader(train_ds, shuffle=True)
            self.val_loader = self._make_loader(val_ds, shuffle=False)
            self.test_loader = self._make_loader(test_ds, shuffle=False)
            self.logger.info(
                f"[Train/Test] Total={n}, Train={len(train_ds)}, "
                f"Val={len(val_ds)}, Test={len(test_ds)}"
            )

        else:
            raise ValueError(f"Unknown evaluation_type: {eval_type}")

        self.logger.info(f"Train batches per epoch: {len(self.train_loader)}")

    # ------------------------------------------------------------------
    # DataLoader factory
    # ------------------------------------------------------------------

    def _make_loader(self, dataset, shuffle: bool, drop_last: bool = False) -> DataLoader:
        opt = self.cfg["optim_args"]
        return DataLoader(
            dataset,
            batch_size=opt["batch_size"],
            shuffle=shuffle,
            num_workers=opt["num_workers"],
            pin_memory=True,
            drop_last=drop_last,
        )

    @staticmethod
    def _subsample(subset: Subset, frac: float) -> Subset:
        n = max(1, int(len(subset) * frac))
        indices = subset.indices[:n]
        return Subset(subset.dataset, indices)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        start = time.time()

        for batch in self.train_loader:
            x = batch["x"].to(self.device, non_blocking=True).float()
            y = batch["y"].to(self.device, non_blocking=True).float()
            self.model.zero_grad()
            loss = self.model(x, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        if self.scheduler is not None and epoch >= self.warm_up_epochs:
            self.scheduler.step()

        avg_loss = total_loss / max(1, len(self.train_loader))

        if epoch % self.cfg["logging_args"]["print_freq"] == 0:
            lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info(
                f"[Epoch {epoch:03d}] loss={avg_loss:.6f}, lr={lr:.6f}, time={time.time()-start:.2f}s"
            )
        return avg_loss

    def train(self) -> dict:
        best_path, best_result = None, None

        for epoch in range(self.epochs):
            self.train_one_epoch(epoch)
            _, result = self.validate(self.val_loader)
            should_stop, improved = self.early_stopper.step(result["f1"])

            if improved:
                best_result = result
                best_path = os.path.join(self.output_dir, "encoder_best.pt")
                torch.save(self.model.state_dict(), best_path)
                self.logger.info(f"New best at epoch {epoch} — F1={result['f1']:.6f}")

            if should_stop:
                self.logger.info(f"Early stopping at epoch {epoch}. Best: {self.early_stopper.best:.6f}")
                break

        if best_path:
            self.model.load_state_dict(torch.load(best_path, map_location=self.device))

        self.logger.info(f"Training complete. Best result: {best_result}")
        return best_result

    def validate(self, dataloader: DataLoader, return_cm: bool = False) -> tuple[float, dict]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in dataloader:
                x = batch["x"].to(self.device).float()
                y = batch["y"].to(self.device).float()
                loss, preds = self.model(x, y, return_preds=True)
                total_loss += loss.item()
                y_hat = (torch.sigmoid(preds) >= 0.5).long().view(-1)
                all_preds.append(y_hat.cpu())
                all_labels.append(y.view(-1).long().cpu())

        avg_loss = total_loss / max(1, len(dataloader))
        result = compute_metrics(torch.cat(all_labels), torch.cat(all_preds))

        log_str = (
            f"Validation — loss={avg_loss:.4f}, acc={result['acc']}, f1={result['f1']}, "
            f"recall={result['recall']}, precision={result['precision']}"
        )
        if return_cm:
            log_str += f", conf_mat={result['conf_mat']}"
        self.logger.info(log_str)

        return avg_loss, result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_results(f1_ls: list, acc_ls: list, cfg: dict):
    results_file = cfg["logging_args"]["results_file"]
    header = ["F1_score", "Accuracy", "Pretraining_model_path", "Finetuned_model_path"]
    row = [np.mean(f1_ls), np.mean(acc_ls), cfg["model_args"]["model_path"], cfg["output_dir"]]
    file_exists = os.path.isfile(results_file)
    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to JSON config file.")
    args = parser.parse_args()

    with open(args.config_path) as f:
        cfg = json.load(f)

    seeds = (
        cfg["seeds"] if isinstance(cfg.get("seeds"), list) and cfg["seeds"]
        else [cfg["seed"]] if isinstance(cfg.get("seed"), int)
        else [42]
    )

    eval_type = cfg["dataset_args"]["evaluation_type"]
    out_dir = create_timestamped_subdir(
        cfg["logging_args"]["base_output_dir"],
        exp_name=cfg.get("exp_name", "exp"),
    )
    f1_ls, acc_ls = [], []

    print(f"Runs: {len(seeds)}, seeds: {seeds}")

    # ── LOSO ──────────────────────────────────────────────────────────
    if eval_type == "loso":
        split_fold = cfg["split_path"]
        folds = sorted(p for p in os.listdir(split_fold) if p.endswith(".csv"))

        for seed in seeds:
            for run_idx, fold in enumerate(folds):
                split_file = os.path.join(split_fold, fold)
                cfg_run = copy.deepcopy(cfg)
                cfg_run.update({"seed": int(seed), "run_num": run_idx, "output_dir": out_dir})
                for split in ("train_dataset_args", "val_dataset_args", "test_dataset_args"):
                    cfg_run["dataset_args"][split]["split_file"] = split_file

                logger = setup_logger(out_dir)
                logger.info(f"=== seed={seed} | fold={fold} ===")

                result = Trainer(cfg_run, logger, run_idx).train()
                f1_ls.append(result["f1"])
                acc_ls.append(result["acc"])

                # Append per-fold results
                csv_path = os.path.join(out_dir, "loso_results.csv")
                pd.DataFrame({"f1_scores": [result["f1"]], "accuracy": [result["acc"]], "split_fold": [fold]}).to_csv(
                    csv_path, mode="a", header=not os.path.isfile(csv_path), index=False
                )

        print(f"LOSO — Mean F1={np.mean(f1_ls):.4f}, Mean Acc={np.mean(acc_ls):.4f}")
        save_results(f1_ls, acc_ls, cfg)

    # ── Train / Test ──────────────────────────────────────────────────
    elif eval_type == "train_test":
        for run_idx, seed in enumerate(seeds):
            cfg_run = copy.deepcopy(cfg)
            cfg_run.update({"seed": int(seed), "run_num": run_idx, "output_dir": out_dir})

            logger = setup_logger(out_dir)
            logger.info(f"=== Run {run_idx+1}/{len(seeds)} | seed={seed} ===")

            trainer = Trainer(cfg_run, logger, run_idx)
            trainer.train()
            _, result = trainer.validate(trainer.test_loader, return_cm=True)
            f1_ls.append(result["f1"])
            acc_ls.append(result["acc"])

    else:
        raise ValueError(f"Unknown evaluation_type: {eval_type}")

    print(f"Mean F1={np.mean(f1_ls):.4f}, Mean Acc={np.mean(acc_ls):.4f}")


if __name__ == "__main__":
    main()