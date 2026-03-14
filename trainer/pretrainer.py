import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.distributed import DistributedSampler

from trainer.trainer import Trainer
from trainer.utils import  save_config_file, get_dataset, set_seed

class PreTrainer(Trainer):
    def __init__(self, cfg: dict, logger, device, rank: int = 0, world_size: int = 1, seed: int = 1, fold:int = 1):
        super().__init__(cfg, logger, device, rank, world_size, seed, fold)
        self.output = {"best_path": None,
                       "best_loss": None,
                       "best_epoch": None}
        self.optim_args = cfg['pretrain_args']['optim_args']
        self.pretrain_output_dir = self.cfg["logging_args"]["pretrain_output_dir"]
        self._build_dataloader()
        self._build_model(cfg['pretrain_args'])
        self._wrap_ddp()
        self._build_optimizer()
        self._build_early_stopper()
      

        if self.rank == 0:
            save_config_file(cfg, self.pretrain_output_dir)

    def _build_dataloader(self):
            ds_args = self.cfg['pretrain_args']["dataset_args"].copy()
            train_ds = get_dataset( ds_args["train_dataset_args"])
            self.cfg['pretrain_args']["dataset_args"]['num_subjects'] = train_ds.num_subjects
            self.train_sampler = DistributedSampler(
                train_ds, num_replicas=self.world_size, rank=self.rank,
                shuffle=True, drop_last=True)
            
            self.train_loader = self._make_loader(train_ds, shuffle=False, drop_last=True, sampler=self.train_sampler)
    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------


    def _build_optimizer(self):
        self.epochs = int(self.optim_args["epochs"])

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.optim_args["lr"],
            betas=(self.optim_args.get("adam_beta1", 0.9), self.optim_args.get("adam_beta2", 0.999)),
            eps=self.optim_args.get("adam_epsilon", 1e-8),
            weight_decay=self.optim_args.get("weight_decay", 0.0),
        )

        self.warm_up_epochs = None
        if self.optim_args.get("use_lr_scheduler", False):
            self.warm_up_epochs = self.optim_args.get("warm_up", None)
            if self.warm_up_epochs is None and self.rank == 0:
                self.logger.warning("'warm_up' not specified in optimizer config.")
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs - (self.warm_up_epochs or 0),
                eta_min=self.optim_args["min_lr"],
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
                self.early_stopper.step(avg_losses["total_loss"]) if self.rank == 0
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
                self.logger.info(f"Best model logged {avg_losses['total_loss']}")
                print(f"Best model logged {avg_losses['total_loss']}")
                self.output['best_loss'] = avg_losses["total_loss"]
                self.output['best_epoch'] = epoch
                self.output['best_path'] = os.path.join(self.pretrain_output_dir, f"encoder_best_{self.fold}.pt")
                self._save_checkpoint(self.output['best_path'])

            if should_stop and self.rank == 0:
                self.logger.info(f"Early stopping at epoch {epoch}. Best: {self.early_stopper.best:.6f}")
                break

        if self.rank == 0:
            self.logger.info("Training complete.")
            self.logger.info(f"Best checkpoint: {self.output['best_path']}")

        return self.output
    
