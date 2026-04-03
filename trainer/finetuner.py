import os
import torch
import torch.optim as optim
import torch.distributed as dist
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.distributed import DistributedSampler

from trainer.trainer import Trainer
from trainer.utils import (compute_metrics, get_dataset, set_seed, save_config_file)


class Finetuner(Trainer):
    def __init__(self, cfg, logger, device, rank=0, world_size=1, seed=1, fold=1):
        super().__init__(cfg, logger, device, rank, world_size, seed, fold)

        self.finetune_output_dir = cfg['logging_args']["finetune_output_dir"]
        self.optim_args = cfg['finetune_args']['optim_args']

        log_cfg = cfg["logging_args"]
        self.print_freq = int(log_cfg["print_freq"])
        self.save_freq  = int(log_cfg["save_freq"])

        # Initialise output dict so best_path always exists
        self.output = {
            'best_f1':   0.0,
            'best_acc':  0.0,
            'best_loss': None,
            'best_epoch': None,
            'best_path': None,
        }

        self._build_dataloader()
        self._build_model(cfg['finetune_args'])
        self._build_optimizer()
        self._wrap_ddp()
        self._build_early_stopper()
        self.logger.info(f"Model path: {self.cfg['finetune_args']['model_args']['model_path']}")
        save_config_file(cfg, self.finetune_output_dir)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_optimizer(self):
        self.epochs = int(self.optim_args["epochs"])
        classifier_params, encoder_params = self.model.get_parameters()
        self.optimizer = optim.AdamW(
            [
                {"params": encoder_params,    "lr": self.optim_args["lr"]},
                {"params": classifier_params, "lr": self.optim_args["lr"]},
            ],
            lr=self.optim_args["lr"],
            weight_decay=self.optim_args.get("weight_decay", 0.0),
            betas=(self.optim_args.get("adam_beta1", 0.9),
                   self.optim_args.get("adam_beta2", 0.999)),
            eps=self.optim_args.get("adam_epsilon", 1e-8),
        )

        self.warm_up_epochs = None
        if self.optim_args.get("use_lr_scheduler", False):
            self.warm_up_epochs = self.optim_args.get("warm_up")
            if self.warm_up_epochs is None:
                self.logger.warning("'warm_up' not specified in optimizer config.")
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs - (self.warm_up_epochs or 0),
                eta_min=self.optim_args["min_lr"],
            )
        else:
            self.scheduler = None

    def _build_dataloader(self):
        ds_args = self.cfg['finetune_args']["dataset_args"]
        train_ds = get_dataset(ds_args["train_dataset_args"])
        print(f"Train dataset size: {len(train_ds)}")
        val_ds   = get_dataset(ds_args["val_dataset_args"])
        test_ds  = get_dataset(ds_args["test_dataset_args"])
        self.train_sampler = DistributedSampler(
            train_ds, num_replicas=self.world_size, rank=self.rank,
            shuffle=True, drop_last=True)
        self.train_loader = self._make_loader(train_ds, shuffle=False,  drop_last=True,  sampler=self.train_sampler)
        self.val_loader   = self._make_loader(val_ds,   shuffle=False, drop_last=False)
        self.test_loader  = self._make_loader(test_ds,  shuffle=False, drop_last=False)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def log_best_model(self, result, epoch=None, avg_loss=None):
        self.logger.info(f"Best model updated → F1={result['f1']:.4f}")
        self.output['best_f1']  = result["f1"]
        self.output['best_acc'] = result["acc"]
        if avg_loss  is not None: self.output['best_loss']  = avg_loss["total_loss"]
        if epoch     is not None: self.output['best_epoch'] = epoch
        self.output['best_path'] = os.path.join(
            self.finetune_output_dir, f"finetuned_best_{self.fold}.pt"
        )
        self._save_checkpoint(self.output['best_path'])

    def train(self):
        for epoch in range(self.epochs):
            
            avg_losses = self.train_one_epoch(epoch)
            _, result  = self.validate(self.val_loader)

            should_stop, improved = self.early_stopper.step(result["f1"])

            # Only broadcast if actually running distributed
            if self.distributed:
                flags = torch.tensor(
                    [int(should_stop), int(improved)],
                    device=self.device, dtype=torch.int64
                )
                dist.broadcast(flags, src=0)
                should_stop = bool(flags[0].item())
                improved    = bool(flags[1].item())

            if improved:
                self.log_best_model(result, epoch, avg_losses)

            if should_stop:
                self.logger.info(
                    f"Early stopping at epoch {epoch}. "
                    f"Best F1: {self.early_stopper.best:.6f}"
                )
                break

        self.logger.info("Finetuning complete.")

        # Cross-dataset: load best checkpoint and run on test set
        val_name  = self.cfg['finetune_args']["dataset_args"]["val_dataset_args"]['data_name']
        test_name = self.cfg['finetune_args']["dataset_args"]["test_dataset_args"]['data_name']

        if val_name != test_name:
            if self.output['best_path'] is None:
                self.logger.warning("No checkpoint saved — skipping test evaluation.")
            else:
                self.logger.info(f"Cross-dataset eval: loading {self.output['best_path']}")
                checkpoint = torch.load(self.output['best_path'], map_location=self.device)
                state_dict = checkpoint['state_dict']

                # Prefix only if model is actually wrapped in DDP
                if self.distributed:
                    state_dict = OrderedDict(
                        ("module." + k, v) for k, v in state_dict.items()
                    )

                self.model.load_state_dict(state_dict)
                _, result = self.validate(self.test_loader)
                self.log_best_model(result)
        else:
            self.logger.info(
                "Val and test datasets are the same — skipping separate test evaluation."
            )

        self.logger.info(f"Best checkpoint: {self.output['best_path']}")
        return self.output

    def validate(self, dataloader, return_cm=False):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in dataloader:
                result = self.model(batch)
                loss   = result["total_loss"]

                if self.distributed:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss /= self.world_size

                total_loss += loss.item()

                y_hat = (torch.sigmoid(result["y_hat"]) >= 0.5).long().view(-1).to(self.device)
                y     = batch["y"].view(-1).long().to(self.device)

                all_preds.append(y_hat)
                all_labels.append(y)

        avg_loss = total_loss / max(1, len(dataloader))
        result   = compute_metrics(torch.cat(all_labels), torch.cat(all_preds))

        self.logger.info(
            f"Validation — loss={avg_loss:.4f}, acc={result['acc']}, "
            f"f1={result['f1']}, recall={result['recall']}, "
            f"precision={result['precision']}"
            + (f", conf_mat={result['conf_mat']}" if return_cm else "")
        )

        return avg_loss, result