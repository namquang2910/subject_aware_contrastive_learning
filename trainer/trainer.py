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

from utils import (
    LossMeter, EarlyStopping, 
    create_model, get_dataset, get_loss,
    set_seed, save_config_file,
)

class Trainer:
    def __init__(self, cfg, logger, device, rank: int = 0, world_size: int = 1):
        self.cfg = cfg
        self.logger = logger
        self.rank = rank
        self.world_size = world_size
        self.distributed = world_size > 1
        self.device = device
        self.output_dir = cfg["output_dir"]
        set_seed(cfg["seed"])
        
        self.output = {"best_path": None,
                       "best_loss": None,
                       "best_epoch": None}
        self.output_dir = self.cfg["logging_args"]["results_file"]
        self.print_freq = int(cfg["logging_args"]["print_freq"])
        self.save_freq = int(cfg["logging_args"]["save_freq"])

        save_config_file(cfg, self.output_dir)

    def _build_model(self):
        loss_fn = get_loss(self.cfg["loss"]["name"], self.cfg["loss"]["loss_args"])
        self.model = create_model(self.cfg, self.device)
        self.model.set_loss_fn(loss_fn)
        self.model.to(self.device)

        if self.distributed:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                output_device=self.device.index if self.device.type == "cuda" else None,
            )

        if self.rank == 0:
            inner = self.model.module if hasattr(self.model, "module") else self.model
            self.logger.info(f"Model.forward signature: {inspect.signature(inner.forward)}")

    def _build_early_stopper(self):
        opt = self.cfg["optim_args"]
        self.early_stopper = EarlyStopping(
            min_delta=opt.get("min_delta", 1e-4),
            patience=opt.get("patience", 10),
            enabled=opt.get("apply_early_stopping", True) if self.rank == 0 else False,
            is_higher=self.cfg.get("early_stopping_higher_better", False),
        )

    def _make_loader(self, dataset, shuffle: bool, drop_last: bool = False,sampler=None) -> DataLoader:
        return DataLoader(
            dataset, batch_size=self.cfg["dataset_args"]["batch_size"],
            shuffle=shuffle, drop_last=drop_last, sampler=sampler,
            num_workers=self.cfg["dataset_args"].get("num_workers", 0)
        )
    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_checkpoint(self, path: str):
        model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save({
            "model": model,
            "state_dict": model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        self.train_sampler.set_epoch(epoch)

        meter = LossMeter()
        start = time.time()

        for _, data in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            result = self.model(data, return_loss=True)
            result["loss"].backward()
            self.optimizer.step()
            meter.update(result)

        avg_losses = meter.average()

        # Sync total loss across ranks
        if self.distributed:
            t = avg_losses["loss"].clone().to(self.device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            avg_losses["loss"] = (t / self.world_size).item()

        if self.scheduler is not None and (self.warm_up_epochs is None or epoch >= self.warm_up_epochs):
            self.scheduler.step()

        if self.rank == 0 and epoch % self.print_freq == 0:
            lr = self.optimizer.param_groups[0]["lr"]
            loss_str = ", ".join(f"{k}={v:.6f}" for k, v in avg_losses.items())
            self.logger.info(f"[Epoch {epoch:03d}] {loss_str}, lr={lr:.6f}, time={time.time()-start:.2f}s")

        return avg_losses