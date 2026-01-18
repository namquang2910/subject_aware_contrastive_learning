"""
Basic model that consists of encoder and classifier.
"""
import os
import torch
import torch.nn as nn

from models.model import Model
from models.utils import get_base_encoder


class EncoderClassifierModel(Model):
    """
    Base class for model that consists of encoder + classifier.
    """
    def __init__(self, base_encoder = None, num_class = 1, subject_invariant = False,
                 model_path = None, freeze_encoder=False, add_projection_head=False, projection_dim=128, device=None):
        """

        :param encoder_name: name of network to use for mapping from raw data to learned representations (h)
        :param encoder_args: arguments for encoder network
        :param classifier_name: name of network for classifier
        :param classifier_args: arguments for classifier network
        :param encoder_state_dict (optional): path to Pytorch state dict that contains weights to initialize model with
        :param freeze_encoder: if True, freeze weights of encoder model
        :param classifier_state_dict (optional): path ot Pytorch state dict w/ weights for classifier model
        :param merge_wrists: if True, x is expected to be concatenation of L & R wrist embeddings
        """
        super().__init__(device=device)
        self.encoder = base_encoder 
        self.num_class = num_class
        self.subject_invariant = subject_invariant
        self.output_dim = num_class
        self.add_projection_head = add_projection_head
        #Freeze encoder if used for linear probe else fine tune
        self._check_freeze_model(freeze_encoder)
        # init encoder and classifier
        self._load_model(model_path= model_path)
        if add_projection_head:
            self.classifier = nn.Sequential(
                nn.Linear(self.encoder.output_dim, projection_dim),
                nn.BatchNorm1d(projection_dim),
                nn.ReLU(),
                nn.Linear(projection_dim, projection_dim),
            )
            self._load_proj_model(model_path=model_path)
            in_dim = self.classifier[-1].in_features   # == projection_dim
            self.classifier[-1] = nn.Linear(in_dim, num_class)
        else:
            self.classifier = nn.Linear(self.encoder.output_dim, num_class)  

    def _load_proj_model(self, model_path=None):
        full_state_dict = torch.load(model_path, map_location=self.device)
        proj_inv_state_dict = {
            k[len("proj_inv."):]: v
            for k, v in full_state_dict.items()
            if k.startswith("proj_inv.")
        }
        self.classifier.load_state_dict(proj_inv_state_dict)

    def _check_output_dim(self):
        if self.num_class != 1:
            raise ("Please specifc the number of class")
        else: 
            print("Performing the binary classification")

    def _check_freeze_model(self, freeze_encoder=False):
        print("Freeze encoder:", freeze_encoder)
        if freeze_encoder:
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False
        else:
            for param in self.encoder.parameters():
                param.requires_grad = True

    def get_parameters(self):
        """
        No parameters
        Retunr classifer parameter and encoder parameters respectively.
        """
        encoder_params = list(self.encoder.parameters())
        fc_params = list(self.classifier.parameters())
        return fc_params, encoder_params

    def forward(self, x, y, return_preds=False):
        self._check_loss_fn()  # confirm that loss function has been set

        x = x.unsqueeze(1).float()  # add channel dim
        h = self.encoder(x)
        y_hat = self.classifier(h)
        y = self._prepare_targets(y)
        loss = self.loss_fn(y_hat, y)
        if return_preds:
            return loss, y_hat
        return loss
    
    def check_grad_status(self):
        """
        Print which parameters are trainable (requires_grad=True)
        and which are frozen (requires_grad=False).
        """
        print("\n=== Gradient Status of Model Parameters ===")
        for name, param in self.encoder.named_parameters():
            status = "TRAINABLE" if param.requires_grad else "FROZEN"
            print(f"{name:40s} | {str(list(param.shape)):20s} | {status}")
        print("==========================================\n")
