"""
Utility functions to be used by neural networks.
"""
from models.net.DeepCNN import DeepCNN
from models.net.CNNEncoder import CNNEncoder
from models.net.moe_encoder import MoEDualBranchEncoder
from models.net.convnextv1 import convnext_stiny
from models.net.moe_n_encoder import MoENExpertEncoder
from models.net.seperate_encoder import SeperateDualBranchEncoder
from models.net.mmoe_n_encoder import MMoENExpertEncoder


def get_base_encoder(name, args):
    if name == 'cnn':
        return CNNEncoder(**args)
    elif name == 'deepcnn':
        return DeepCNN(**args)
    elif name == 'convnextv1':
        return convnext_stiny(**args)
    elif name == 'moe':
        return MoEDualBranchEncoder(**args)
    elif name == 'moe_n':
        return MoENExpertEncoder(**args)
    elif name == "seperate_encoder":
        return SeperateDualBranchEncoder(**args)
    elif name == "mmoe_n":
        return MMoENExpertEncoder(**args)