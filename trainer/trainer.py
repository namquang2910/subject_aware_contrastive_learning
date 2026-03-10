import time
import inspect
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from trainer.utils import (
    LossMeter, EarlyStopping, 
    create_model, get_loss,
    set_seed, save_config_file, get_dataset
)

class Trainer:
    def __init__(self, cfg, logger, device, rank: int = 0, world_size: int = 1, seed:int = 1, fold: int = None):
        self.cfg = cfg
        self.logger = logger
        self.rank = rank
        self.world_size = world_size
        self.distributed = world_size > 1
        self.fold = fold
        self.device = device
        self.optim_args = None
        self.output_dir = cfg["logging_args"]["output_dir"]
        set_seed(seed)
        
        self.output = {"best_path": None,
                       "best_loss": None,
                       "best_epoch": None}
        self.results_file = self.cfg["logging_args"]["results_file"]
        self.print_freq = int(cfg["logging_args"]["print_freq"])
        self.save_freq = int(cfg["logging_args"]["save_freq"])

        save_config_file(cfg, self.output_dir)

    def _build_model(self, training_cfg):
        """
        training_cfg: specific the training config by passing the pretrain_args or the finetune_args
        """
        loss_fn = get_loss(training_cfg["loss"]["name"], training_cfg["loss"]["loss_args"])
        self.model = create_model(training_cfg, self.device)
        self.model.set_loss_fn(loss_fn)
        self.model.to(self.device)

    def _wrap_ddp(self):
        """
        Wrap self.model with DistributedDataParallel safely.
        Assumes torch.distributed is already initialized.
        """

        if not self.distributed:
            return

        # Ensure model is on the correct device BEFORE wrapping
        self.model.to(self.device)

        # Convert BatchNorm → SyncBatchNorm (only for multi-GPU training)
        if self.device.type == "cuda":
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        # Wrap with DDP
        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.device.index] if self.device.type == "cuda" else None,
            output_device=self.device.index if self.device.type == "cuda" else None,
            find_unused_parameters=False,   # change to True only if needed
            broadcast_buffers=True,
        )
    def _build_early_stopper(self):
        self.early_stopper = EarlyStopping(
            min_delta=self.optim_args.get("min_delta", 1e-4),
            patience=self.optim_args.get("patience", 10),
            enabled=self.optim_args.get("apply_early_stopping", True) if self.rank == 0 else False,
            is_higher=self.optim_args.get("early_stopping_higher_better", False),
        )
    

    def _make_loader(self, dataset, shuffle: bool, drop_last: bool = False,sampler=None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.optim_args["batch_size"],
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.optim_args["num_workers"],
            pin_memory=True,
            drop_last=drop_last,
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
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        meter = LossMeter()
        start = time.time()

        for _, data in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            result = self.model(data)
            result["total_loss"].backward()
            self.optimizer.step()
            result.pop("y_hat", None) #remove y_hat
            meter.update(result)

        avg_losses = meter.average()

        # Sync total loss across ranks
        if self.distributed:
            t = avg_losses["total_loss"].clone().to(self.device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            avg_losses["total_loss"] = (t / self.world_size).item()

        if self.scheduler is not None and (self.warm_up_epochs is None or epoch >= self.warm_up_epochs):
            self.scheduler.step()

        if self.rank == 0 and epoch % self.print_freq == 0:
            lr = self.optimizer.param_groups[0]["lr"]
            loss_str = ", ".join(f"{k}={v:.6f}" for k, v in avg_losses.items())
            self.logger.info(f"[Epoch {epoch:03d}] {loss_str}, lr={lr:.6f}, time={time.time()-start:.2f}s")

        return avg_losses