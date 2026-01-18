import os
import csv
import json
import time
import copy
import argparse
import inspect
import math

from tqdm import tqdm
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

from loss.cl_loss import NCELoss
#from loss.cl_dtw import NCE_DTWLoss
from models.utils import get_base_encoder
from utils import (
    set_seed,
    create_timestamped_subdir,
    save_config_file,
    setup_logger,
    EarlyStopping,
)
from models import pretrain_builder, subject_invariant_pretrain, subject_specific_pretrain, subject_aware_pretrain, invariant_model
from datasets.wesad_dataset import WESADDataset
from datasets.psy_dataset import PsyDataset
from datasets.swell_dataset import SWELLDataset

class Trainer:
    def __init__(self, cfg, logger, num_class, train_loader, train_sampler, device, rank=0, world_size=1):
        self.train_loader = train_loader
        self.train_sampler = train_sampler
        self.logger = logger
        self.rank = rank
        self.world_size = world_size
        self.distributed = world_size > 1
        self.device = device
        self.best_total_loss = 0.0
        self.best_subject_loss = 0.0
        self.best_contrastive_loss = 0.0
        self.output_dir = cfg["output_dir"]
        self.cfg = cfg
        # --- Seed ---
        set_seed(cfg["seed"])

        # --- Loss ---
        self.loss_fn = get_loss(cfg['loss']['name'], cfg['loss']['loss_args'])

        # --- Model ---
        enc_name = cfg["model_args"]["base_encoder"]
        enc_args = cfg["model_args"]["base_encoder_args"]
        encoder = get_base_encoder(enc_name, enc_args)

        projection_output = cfg["model_args"].get("projection_output", 32)
        use_softdtw = cfg["model_args"].get("use_softdtw", False)
        model_type = cfg["model_args"].get("model_type", "contrastive")

        if model_type == "contrastive":
            self.model = pretrain_builder.ContrastiveModel(
                encoder, projection_output=projection_output,
                device=self.device,use_softdtw = use_softdtw
            )
        elif model_type == "subject_invariant":
            num_subjects = cfg["model_args"].get("num_subjects", num_class)
            grl_lambda = cfg["model_args"].get("grl_lambda", 1.0)
            self.model = subject_invariant_pretrain.AdversarialContrastiveModel(
                encoder,
                projection_output=projection_output,
                num_subjects=num_subjects,
                grl_lambda=grl_lambda,
                device=self.device
            )
        elif model_type == "subject_specific":
            self.model = subject_specific_pretrain.SubjectSpecificContrastiveModel(
                encoder, projection_output=projection_output,
                device=self.device
            )
        elif model_type == "subject_aware":
            num_subjects = cfg["model_args"].get("num_subjects", num_class)
            grl_lambda = cfg["model_args"].get("grl_lambda", 1.0)
            self.model = subject_aware_pretrain.SubjectAwareContrastiveModel(
                encoder,
                projection_output=projection_output,
                device=self.device,
                num_subjects=num_subjects,
                grl_lambda=grl_lambda
            )
        elif model_type == "invariant_model":
            num_subjects = cfg["model_args"].get("num_subjects", num_class)
            grl_lambda = cfg["model_args"].get("grl_lambda", 1.0)
            self.model = invariant_model.InvariantContrastiveModel(
                encoder,
                projection_output=projection_output,
                num_subjects=num_subjects,
                grl_lambda=grl_lambda,
                device=self.device
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.model.set_loss_fn(self.loss_fn)
        self.model.to(self.device)

        # Wrap with DDP if distributed
        if self.distributed:
            # Convert all BatchNorm layers to SyncBatchNorm
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                output_device=self.device.index if self.device.type == "cuda" else None,
            )

        if self.rank == 0:
            self.logger.info(
                f"Model.forward signature: {inspect.signature(self.model.module.forward if hasattr(self.model, 'module') else self.model.forward)}"
            )

        # --- Optimizer & Scheduler ---
        opt_cfg = cfg["optim_args"]
        lr_value = opt_cfg["lr"]
        min_lr_value = opt_cfg["min_lr"]
        self.epochs = int(opt_cfg["epochs"])

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr_value,
            betas=(
                opt_cfg.get("adam_beta1", 0.9),
                opt_cfg.get("adam_beta2", 0.999),
            ),
            eps=opt_cfg.get("adam_epsilon", 1e-8),
            weight_decay=opt_cfg.get("weight_decay", 0.0),
        )

        self.use_lr_scheduler = opt_cfg.get("use_lr_scheduler", False)
        if self.use_lr_scheduler:
            self.warm_up_epochs = opt_cfg.get("warm_up", None)
            if self.warm_up_epochs is None and self.rank == 0:
                self.logger.warning("'warm_up' not specified in optimizer config")

            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs - (self.warm_up_epochs or 0),
                eta_min=min_lr_value,
            )
        else:
            self.scheduler = None

        # --- Early Stopping ---
        self.early_stopper = EarlyStopping(
            min_delta=opt_cfg.get("min_delta", 1e-4),
            patience=opt_cfg.get("patience", 10),
            enabled=opt_cfg.get("apply_early_stopping", True) if self.rank == 0 else False,
            is_higher= cfg.get("early_stopping_higher_better", False)
        )

        # --- Logging / Output ---
        log_cfg = cfg["logging_args"]
        self.print_freq = int(log_cfg["print_freq"])
        self.save_freq = int(log_cfg["save_freq"])

        if self.rank == 0:
            save_config_file(cfg, self.output_dir)

    def train_one_epoch(self, epoch):
        start_time = time.time()
        self.model.train()
        losses = 0.0
        sub_losses = 0.0
        cl_losses = 0.0
        dist_losses = 0.0
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        for batch_idx, data in enumerate(self.train_loader):
            self.model.zero_grad()
            loss, sub_loss, cl_loss, distill_loss = self.model(data, return_loss=True)
            loss.backward()
            self.optimizer.step()
            losses += loss.detach()
            sub_losses += sub_loss.detach()
            cl_losses += cl_loss.detach()
            dist_losses += distill_loss.detach()
        # Compute average loss across all ranks
        avg_loss = losses / max(1, len(self.train_loader))
        avg_loss_tensor = avg_loss.detach().clone().to(self.device)
        avg_sub_loss = sub_losses / max(1, len(self.train_loader))
        avg_cl_loss = cl_losses / max(1, len(self.train_loader))
        avg_dist_loss = dist_losses / max(1, len(self.train_loader))

        if self.distributed:
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss_tensor /= self.world_size

        avg_loss = avg_loss_tensor.item()

        if self.scheduler is not None and (
            self.warm_up_epochs is None or epoch >= self.warm_up_epochs
        ):
            self.scheduler.step()

        elapsed = time.time() - start_time

        if self.rank == 0 and (epoch % self.print_freq) == 0:
            last_lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info(
                f"[Epoch {epoch:03d}] loss={avg_loss:.6f}, sub_loss={avg_sub_loss:.6f}, cl_loss={avg_cl_loss:.6f}, dist_loss={avg_dist_loss:.6f}, lr={last_lr:.6f} time={elapsed:.2f}s"
            )

        return avg_loss, avg_sub_loss, avg_cl_loss

    def train(self):
        e_best_path = None
        if self.rank == 0:
            print(self.model)

        for epoch in range(self.epochs):
            train_loss, train_sub_loss, train_cl_loss = self.train_one_epoch(epoch)

            # Early stopping only on rank 0
            if self.rank == 0:
                should_stop, improved = self.early_stopper.step(train_loss)
            else:
                should_stop, improved = False, False

            # Broadcast decisions to all ranks
            if self.distributed:
                flags = torch.tensor(
                    [int(should_stop), int(improved)],
                    device=self.device,
                    dtype=torch.int64,
                )
                dist.broadcast(flags, src=0)
                should_stop = bool(flags[0].item())
                improved = bool(flags[1].item())

            # Checkpointing only on rank 0
            if self.rank == 0:
#                if (epoch % self.save_freq) == 0:
#                    ckpt_path = os.path.join(
#                        self.output_dir, f"encoder_epoch_{epoch}.pt"
#                    )
#                    self.save_checkpoint(ckpt_path)

                if improved:
                    self.best_total_loss = train_loss
                    self.best_subject_loss = train_sub_loss
                    self.best_contrastive_loss = train_cl_loss
                    e_best_path = os.path.join(self.output_dir, "encoder_best.pt")
                    self.save_checkpoint(e_best_path)

                if should_stop:
                    self.logger.info(
                        f"Early stopping at epoch {epoch}. Best loss: {self.early_stopper.best:.6f}"
                    )

            if should_stop:
                break

        if self.rank == 0:
            self.logger.info("Training complete.")
            if e_best_path:
                self.logger.info(f"Best checkpoint: {e_best_path}")
                self.save_file(e_best_path)

    def save_file(self, path):
        header = [
            "Best_pretraining Loss",
            "Best_subject Loss",
            "Best_contrastive Loss",
            "Model_type",
            "Best_model_path"
        ]
        row = [
            self.best_total_loss,
            self.best_subject_loss,
            self.best_contrastive_loss,
            self.cfg["model_args"]["model_type"],
            path
        ]

        file_exists = os.path.isfile(self.cfg["logging_args"]["results_file"])

        with open(self.cfg["logging_args"]["results_file"], 'a', newline='') as fd:
            writer = csv.writer(fd)
            # write header only once
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)
  
    def save_checkpoint(self, filename):
        # unwrap DDP if needed
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        state = {
            "model": model_to_save,
            "state_dict": model_to_save.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, filename)


class DummyLogger:
    """Logger that does nothing on non-zero ranks."""

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to JSON config file.")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        cfg = json.load(f)

    # --- Distributed setup ---
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        world_size = 1
    print(f"World size: {world_size}")
    print(f"Num GPUS Available: {torch.cuda.device_count()}")
    distributed = world_size > 1

    if distributed:
        # These are set automatically by torchrun
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

        print(
            f"Initialized distributed training: rank {rank}/{world_size}, local_rank {local_rank}",
            flush=True,
        )
    else:
        rank = 0
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Running in single-process (non-distributed) mode.", flush=True)


    # --- normalize seeds ---
    if "seeds" in cfg and isinstance(cfg["seeds"], list) and len(cfg["seeds"]) > 0:
        seeds = cfg["seeds"]
    elif "seed" in cfg and isinstance(cfg["seed"], int):
        seeds = [cfg["seed"]]
    else:
        seeds = [42]

    log_cfg = cfg["logging_args"]
    base_out = log_cfg["base_output_dir"]
    exp_name = cfg.get("exp_name", "exp")
    base_out = os.path.join(base_out, cfg["model_args"]["model_type"])
    num_runs = len(seeds)
    if rank == 0:
        print(f"[Runs] num_runs={num_runs}, seeds={seeds}")

    # run once per seed
    for run_idx, seed in enumerate(seeds):
        cfg_run = copy.deepcopy(cfg)
        cfg_run["seed"] = int(seed)
        cfg_run["run_num"] = run_idx

        #Update the experiment name of the config file 
        if rank == 0:
            out_dir = create_timestamped_subdir(base_out, exp_name=exp_name, dataset = cfg_run["dataset_args"]['data_name'])
        else:
            out_dir = None

        if distributed:
            obj = [out_dir]
            dist.broadcast_object_list(obj, src=0)
            out_dir = obj[0]

        cfg_run["output_dir"] = out_dir

        # Setup logger
        if rank == 0:
            logger = setup_logger(out_dir)
            logger.info(f"=== Run {run_idx+1}/{num_runs} | seed={seed} ===")
            print(f"[Run] run_idx={run_idx + 1}/{num_runs}, seed={seed}")
            logger.info(f"Output dir: {out_dir}")
        else:
            logger = DummyLogger()

        # --- Dataloader ---
        ds_args = cfg_run["dataset_args"]["train_dataset"]
        print("Initalize the dataset")
        train_ds = get_dataset(cfg_run["dataset_args"]['data_name'], ds_args)

        train_sampler = DistributedSampler(
                train_ds,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg_run["optim_args"]["batch_size"],
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=cfg_run["optim_args"]["num_workers"],
            pin_memory=True,
            drop_last=True,
        )
        num_class = train_ds.num_subjects
        # Trainer per rank
        trainer = Trainer(cfg_run, logger=logger, num_class = num_class,
                          train_loader=train_loader, train_sampler = train_sampler,
                          device=device, rank=rank, world_size=world_size)
        trainer.train()

    if distributed:
        dist.destroy_process_group()

def get_dataset(data_name, ds_args):
    if data_name =="WESADDataset":
        return WESADDataset(**ds_args)
    elif data_name == "PsychioNet":
        return PsyDataset(**ds_args)
    elif data_name == "SWELLDataset":
        return SWELLDataset(**ds_args)

def get_loss(name, loss_args):
    if name == "NCE":
        return NCELoss(**loss_args)

if __name__ == "__main__":
    main()
