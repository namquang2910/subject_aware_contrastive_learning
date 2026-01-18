"""
Utility functions to be used by neural networks.
"""
from models.net.DeepCNN import DeepCNN
from models.net.CNNEncoder import CNNEncoder
from models.net.Resnet25 import ResNet25
from models.net.convnextv1 import convnext_stiny, convnext_tiny, convnext_small, convnext_base, convnext_large, convnext_xlarge

def get_base_encoder(name, args):
    if name == 'cnn':
        return CNNEncoder(**args)
    elif name == 'deepcnn':
        return DeepCNN(**args)
    elif name == 'resnet':
        return ResNet25(**args)
    elif name == 'convnextv1':
        return convnext_stiny(**args)