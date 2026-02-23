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

from loss.cl_loss import NCELoss
from utils import (
    LossMeter, EarlyStopping,
    create_experiment, create_model,
    get_dataset, save_config_file,
    set_seed, setup_logger, DummyLogger
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_loss(name: str, loss_args: dict):
    if name == "NCE":
        return NCELoss(**loss_args)
    raise ValueError(f"Unknown loss: {name}")


def setup_distributed():
    """Initialize distributed training if WORLD_SIZE is set, else run single-process."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend="nccl")
        print(f"Distributed: rank {rank}/{world_size}, local_rank {local_rank}", flush=True)
    else:
        rank, local_rank = 0, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Single-process mode.", flush=True)

    return rank, world_size, device


def resolve_seeds(cfg: dict) -> list:
    if isinstance(cfg.get("seeds"), list) and cfg["seeds"]:
        return cfg["seeds"]
    if isinstance(cfg.get("seed"), int):
        return [cfg["seed"]]
    return [42]

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, cfg: dict, logger, device, rank: int = 0, world_size: int = 1):
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
        ds_args = self.cfg["dataset_args"]["train_dataset"]
        train_ds = get_dataset(self.cfg["dataset_args"]["data_name"], ds_args)
        self.cfg["dataset"]["num_class"] = train_ds.num_subjects

        self.train_sampler = DistributedSampler(
            train_ds, num_replicas=self.world_size, rank=self.rank,
            shuffle=True, drop_last=True,
        )
        opt = self.cfg["optim_args"]
        self.train_loader = DataLoader(
            train_ds,
            batch_size=opt["batch_size"],
            sampler=self.train_sampler,
            num_workers=opt["num_workers"],
            pin_memory=True,
            drop_last=True,
        )

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

    def _build_optimizer(self):
        opt = self.cfg["optim_args"]
        self.epochs = int(opt["epochs"])

        self.optimizer = optim.Adam(
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

    def _build_early_stopper(self):
        opt = self.cfg["optim_args"]
        self.early_stopper = EarlyStopping(
            min_delta=opt.get("min_delta", 1e-4),
            patience=opt.get("patience", 10),
            enabled=opt.get("apply_early_stopping", True) if self.rank == 0 else False,
            is_higher=self.cfg.get("early_stopping_higher_better", False),
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

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

    def train(self):
        best_path = None

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
                best_path = os.path.join(self.output_dir, "encoder_best.pt")
                self._save_checkpoint(best_path)

            if should_stop:
                self.logger.info(f"Early stopping at epoch {epoch}. Best: {self.early_stopper.best:.6f}")
                break

        if self.rank == 0:
            self.logger.info("Training complete.")
            if best_path:
                self.logger.info(f"Best checkpoint: {best_path}")
                self._save_results(best_path)

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

    def _save_results(self, path: str, loss=None):
        results_file = self.cfg["logging_args"]["results_file"]
        header = ["exp_name", "best_loss", "model_type", "best_model_path"]
        row = [self.cfg.get("exp_name", "exp"), loss, self.cfg["model_args"]["model_type"], path]
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

    rank, world_size, device = setup_distributed()
    seeds = resolve_seeds(cfg)

    base_out = os.path.join(cfg["logging_args"]["base_output_dir"], cfg["model_args"]["model_type"])
    exp_name = cfg.get("exp_name", "exp")

    if rank == 0:
        print(f"Runs: {len(seeds)}, seeds: {seeds}")

    for run_idx, seed in enumerate(seeds):
        cfg_run = copy.deepcopy(cfg)
        cfg_run["seed"] = int(seed)
        cfg_run["run_num"] = run_idx

        # Create output dir on rank 0, then broadcast
        out_dir = (
            create_experiment(base_out, exp_name=exp_name, dataset=cfg_run["dataset_args"]["data_name"])
            if rank == 0 else None
        )
        if world_size > 1:
            obj = [out_dir]
            dist.broadcast_object_list(obj, src=0)
            out_dir = obj[0]
        cfg_run["output_dir"] = out_dir

        if rank == 0:
            logger = setup_logger(out_dir)
            logger.info(f"=== Run {run_idx+1}/{len(seeds)} | seed={seed} | out={out_dir} ===")
        else:
            logger = DummyLogger()

        Trainer(cfg_run, logger=logger, device=device, rank=rank, world_size=world_size).train()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()