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

from trainer import Trainer
from utils import ( get_dataset, set_seed, save_config_file,)

class PreTrainer(Trainer):
    def __init__(self, cfg: dict, logger, device, rank: int = 0, world_size: int = 1):
        super().__init__(cfg, logger, device, rank, world_size)

        set_seed(cfg["seed"])
        self._build_dataloader()
        self._build_model()
        self._build_optimizer()
        self._build_early_stopper()
        self.output = {"best_path": None,
                       "best_loss": None,
                       "best_epoch": None}
        self.output_dir = self.cfg["logging_args"]["results_file"]
        self.print_freq = int(cfg["logging_args"]["print_freq"])
        self.save_freq = int(cfg["logging_args"]["save_freq"])

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

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=opt["lr"],
            betas=(opt.get("adam_beta1", 0.9), opt.get("adam_beta2", 0.999)),
            eps=opt.get("adam_epsilon", 1e-8),
            weight_decay=opt.get("weight_decay", 0.0),
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

            should_stop, improved = (
                self.early_stopper.step(avg_losses["loss"]) if self.rank == 0
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
                
                self.output['best_loss'] = avg_losses["loss"]
                self.output['best_epoch'] = epoch
                self.output['best_path'] = os.path.join(self.output_dir, "encoder_best.pt")
                self._save_checkpoint(self.output['best_path'])

            if should_stop:
                self.logger.info(f"Early stopping at epoch {epoch}. Best: {self.early_stopper.best:.6f}")
                break

        if self.rank == 0:
            self.logger.info("Training complete.")
            self.logger.info(f"Best checkpoint: {self.output['best_path']}")

        return self.output
    
