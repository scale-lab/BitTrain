import torch
from edgify.models.base import BasicBlock, ResNet

__all__ = [
    'resnet18'
]

def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model

def resnet18(**kwargs):
    return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)

