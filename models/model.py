"""
Base class for neural network models.
"""

import os
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Base class for neural network model.
    """
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.loss_fn = None
        
    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    def _check_loss_fn(self):
        if self.loss_fn is None:
            print("Please set loss function")
            exit(1)

    def _load_model(self, model_path):
        if not os.path.isfile(model_path):
            print(f"=> no checkpoint found at '{model_path}'")
            exit(1)

        print(f"=> loading checkpoint '{model_path}'")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]
        new_state = {}
        for k, v in state_dict.items():
            # 1) remove "module." prefix if it exists (DDP checkpoint)
            if k.startswith("module."):
                k = k.replace("module.", "", 1)

            # 2) keep only encoder_q.* layers
            if k.startswith("encoder_q."):
                new_k = k.replace("encoder_q.", "", 1)
                new_state[new_k] = v

            if k.startswith("encoder."):
                new_k = k.replace("encoder.", "", 1)
                new_state[new_k] = v    

            if k.startswith("encoder_i."):
                new_k = k.replace("encoder_i.", "", 1)
                new_state[new_k] = v   
        # Load into your encoder
        msg = self.encoder.load_state_dict(new_state, strict=False)
       # print("Missing keys:", msg.missing_keys)
        if len(msg.missing_keys) != 0:
            print(state_dict.keys())
            print(msg)
            raise ValueError(f"Error loading pretrained encoder.\nMissing keys")

        print(f"=> loaded pre-trained model '{model_path}'")

    def save(self, output_dir, filename):
        # save whole model
        file_path = os.path.join(output_dir, filename)
        torch.save(state_dict, file_path)
        
        print("Model saved to {}".format(file_path))

    def _prepare_targets(self, targets):
        """
        :param targets: prediction targets to prepare to use as argument to loss function
                       (i.e., put on the right device & convert to float if needed)
        :return: processed targets
        """
        if targets.dtype == torch.double:
            # convert to float
            targets = targets.float()
        # make sure target is 2D
        if len(targets.shape) == 1:
            targets = targets[:, None]
            # also need to convert to float if binary classification
            targets = targets.float()
        return targets.to(device=self.device)
    
    def get_parameters():
        raise NotImplementedError