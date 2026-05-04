"""
Utility functions to be used by neural networks.
"""
from models.net.CNNEncoder import CNNEncoder
from models.net.moe_encoder import MoEDualBranchEncoder
from models.net.convnextv1 import convnext_stiny


def get_base_encoder(name, args):
    if name == 'cnn':
        return CNNEncoder(**args)
    elif name == 'convnextv1':
        return convnext_stiny(**args)
    elif name == 'moe':
        return MoEDualBranchEncoder(**args)
