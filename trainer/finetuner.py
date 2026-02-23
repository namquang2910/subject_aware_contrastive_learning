import os
import csv
import copy
import json
import time
import inspect
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

from trainer.trainer import Trainer
from utils import ( compute_metrics, get_dataset, set_seed, save_config_file,)

class Finetuner(Trainer):
    def __init__(self, cfg: dict, logger, device, rank: int = 0, world_size: int = 1):
        super().__init__(cfg, logger, device, rank, world_size)
        self.cfg = cfg
        self.logger = logger
        self.rank = rank
        self.world_size = world_size
        self.distributed = world_size > 1
        self.device = device
        self.output_dir = cfg["output_dir"]

        set_seed(cfg["seed"])
        self._build_dataloader()
        self._build_model()
        self._build_optimizer()
        self._build_early_stopper()

        log_cfg = cfg["logging_args"]
        self.print_freq = int(log_cfg["print_freq"])
        self.save_freq = int(log_cfg["save_freq"])

        if self.rank == 0:
            save_config_file(cfg, self.output_dir)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_dataloader(self):
        ds_args = self.cfg["dataset_args"]
        train_ds = get_dataset(self.cfg["dataset_args"]["data_name"], ds_args["train_dataset"])
        val_ds = get_dataset(self.cfg["dataset_args"]["data_name"], ds_args["val_dataset"])
        test_ds = get_dataset(self.cfg["dataset_args"]["data_name"], ds_args["test_dataset"])
        self.cfg["dataset"]["num_class"] = train_ds.num_subjects

        self.train_sampler = DistributedSampler(
            train_ds, num_replicas=self.world_size, rank=self.rank,
            shuffle=True, drop_last=True)
        
        self.train_loader = self._make_loader(train_ds, shuffle=False, drop_last=True, sampler=self.train_sampler)
        self.val_loader = self._make_loader(val_ds, shuffle=False, drop_last=False)
        self.test_loader = self._make_loader(test_ds, shuffle=False, drop_last=False)
        
    def _make_loader(self, dataset, shuffle: bool, drop_last: bool = False,sampler=None) -> DataLoader:
        opt = self.cfg["optim_args"]
        return DataLoader(
            dataset,
            batch_size=opt["batch_size"],
            shuffle=shuffle,
            sampler=sampler,
            num_workers=opt["num_workers"],
            pin_memory=True,
            drop_last=drop_last,
        )

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

        self.warm_up_epochs = None
        if opt.get("use_lr_scheduler", False):
            self.warm_up_epochs = opt.get("warm_up")
            if self.warm_up_epochs is None and self.rank == 0:
                self.logger.warning("'warm_up' not specified in optimizer config.")
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs - (self.warm_up_epochs or 0),
                eta_min=opt["min_lr"],
            )
        else:
            self.scheduler = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self):
        for epoch in range(self.epochs):
            avg_losses = self.train_one_epoch(epoch)
            _, result = self.validate(self.val_loader)

            should_stop, improved = (
                self.early_stopper.step(result["f1"]) if self.rank == 0
                else (False, False)
            )

            # Broadcast early-stopping decisions to all ranks
            if self.distributed:
                flags = torch.tensor(
                    [int(should_stop), int(improved)], device=self.device, dtype=torch.int64
                )
                dist.broadcast(flags, src=0)
                should_stop, improved = bool(flags[0].item()), bool(flags[1].item())

            if self.rank == 0 and improved:
                self.output['best_f1'] = result["f1"]
                self.output['best_acc'] = result["acc"]
                self.output['best_loss'] = avg_losses["loss"]
                self.output['best_epoch'] = epoch
                self.output['best_path'] = os.path.join(self.output_dir, f"finetuned_best_{self.cfg['fold']}.pt")
                self._save_checkpoint(self.output['best_path'])

            if should_stop:
                self.logger.info(f"Early stopping at epoch {epoch}. Best: {self.early_stopper.best:.6f}")
                break

        if self.rank == 0:
            self.logger.info("Training complete.")
            self.logger.info(f"Best checkpoint: {self.output['best_path']}")


    def validate(self, dataloader: DataLoader, return_cm: bool = False) -> tuple[float, dict]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in dataloader:
                result = self.model(batch, return_preds=True)
                total_loss += result["total_loss"].item()
                y_hat = (torch.sigmoid(result["y_hat"]) >= 0.5).long().view(-1)
                all_preds.append(y_hat.cpu())
                all_labels.append(batch["y"].view(-1).long().cpu())
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
