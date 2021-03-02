import pytest
import torch

from edgify import models


class TestResnet:
    def test_forward(self):
        model = models.resnet18()
        data = torch.rand((64, 3, 224, 224))

        out = model(data)

