import torch
from torchvision.models.utils import load_state_dict_from_url
from edgify.models.base import BasicBlock, Bottleneck, ResNet

__all__ = [
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152'
]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def _resnet(arch, block, layers, pretrained, progress, bitmap, **kwargs):
    model = ResNet(block, layers, bitmap, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet18(pretrained=False, progress=True, bitmap=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], 
                    pretrained, progress, bitmap, **kwargs)

def resnet34(pretrained=False, progress=True, bitmap=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], 
                    pretrained, progress, bitmap, **kwargs)

def resnet50(pretrained=False, progress=True, bitmap=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], 
                    pretrained, progress, bitmap, **kwargs)

def resnet101(pretrained=False, progress=True, bitmap=True, **kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], 
                    pretrained, progress, bitmap, **kwargs)

def resnet152(pretrained=False, progress=True, bitmap=True, **kwargs):
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], 
                    pretrained, progress, bitmap, **kwargs)

