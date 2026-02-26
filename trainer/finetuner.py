import os

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from trainer.trainer import Trainer
from trainer.utils import ( compute_metrics, get_dataset, set_seed, save_config_file,)

class Finetuner(Trainer):
    def __init__(self, cfg: dict, logger, device, rank: int = 0, world_size: int = 1, seed: int=1, fold:int = 1):
        super().__init__(cfg, logger, device, rank, world_size, seed, fold)

        self.finetune_output_dir = cfg['logging_args']["finetune_output_dir"]
        self.optim_args = cfg['finetune_args']['optim_args']

        log_cfg = cfg["logging_args"]
        self.print_freq = int(log_cfg["print_freq"])
        self.save_freq = int(log_cfg["save_freq"])
        
        self._build_dataloader(finetune=True)
        self._build_model(cfg['finetune_args'])
        self._build_optimizer()
        self._wrap_ddp()
        self._build_early_stopper()


        if self.rank == 0:
            save_config_file(cfg, self.finetune_output_dir)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_optimizer(self):
        self.epochs = int(self.optim_args["epochs"])
        print(self.model)
        encoder_params, classifier_params = self.model.get_parameters()
        self.optimizer = optim.AdamW(
            [
                {"params": encoder_params, "lr": self.optim_args["lr"] / 10},
                {"params": classifier_params, "lr": self.optim_args["lr"]},
            ],
            lr=self.optim_args["lr"],
            weight_decay=self.optim_args.get("weight_decay", 0.0),
            betas=(self.optim_args.get("adam_beta1", 0.9), self.optim_args.get("adam_beta2", 0.999)),
            eps=self.optim_args.get("adam_epsilon", 1e-8),
        )

        self.warm_up_epochs = None
        if self.optim_args.get("use_lr_scheduler", False):
            self.warm_up_epochs = self.optim_args.get("warm_up")
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
                self.logger.info(f"Best model logged {result['f1']}")
                print(f"Best model logged {result['f1']}")
                self.output['best_f1'] = result["f1"]
                self.output['best_acc'] = result["acc"]
                self.output['best_loss'] = avg_losses["total_loss"]
                self.output['best_epoch'] = epoch
                self.output['best_path'] = os.path.join(self.finetune_output_dir, f"finetuned_best_{self.fold}.pt")
                self._save_checkpoint(self.output['best_path'])

            if should_stop and self.rank == 0:
                self.logger.info(f"Early stopping at epoch {epoch}. Best: {self.early_stopper.best:.6f}")
                break

        if self.rank == 0:
            self.logger.info("Training complete.")
            self.logger.info(f"Best checkpoint: {self.output['best_path']}")
        if self.distributed:
            dist.barrier()
        return self.output

    def validate(self, dataloader, return_cm=False):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in dataloader:
                result = self.model(batch)
                loss = result["total_loss"]

                if self.distributed:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss /= self.world_size
                total_loss += loss.item()

                y_hat = (torch.sigmoid(result["y_hat"]) >= 0.5).long().view(-1)
                y = batch["y"].view(-1).long()

                if self.distributed:
                    y_hat = y_hat.to(self.device)
                    y = y.to(self.device)
                    y_hat = self._gather_tensor(y_hat)
                    y = self._gather_tensor(y)

                all_preds.append(y_hat)   # ← was missing
                all_labels.append(y)      # ← was missing

        avg_loss = total_loss / max(1, len(dataloader))
        result = compute_metrics(torch.cat(all_labels), torch.cat(all_preds))

        if self.rank == 0:
            log_str = (
                f"Validation — loss={avg_loss:.4f}, acc={result['acc']}, "
                f"f1={result['f1']}, recall={result['recall']}, "
                f"precision={result['precision']}"
            )
            if return_cm:
                log_str += f", conf_mat={result['conf_mat']}"
            self.logger.info(log_str)

        return avg_loss, result
    
    def _gather_tensor(self, tensor):
        if not self.distributed:
            return tensor

        tensor = tensor.contiguous()
        tensors = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(tensors, tensor)
        return torch.cat(tensors, dim=0)