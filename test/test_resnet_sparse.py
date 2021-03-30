import pytest
import torch
import time

from edgify import models
from torch.nn.functional import nll_loss

class TestResnetSparse:
    def test_forward(self):
        model = models.resnet18()
        data = torch.rand((8, 3, 224, 224))

        out = model(data)

    def test_backward(self):
        model = models.resnet101(pretrained=True)
        inputs = torch.rand((8, 3, 224, 224))
        labels = torch.randint(2, (8, ))

        out = model(inputs)
        loss = nll_loss(out, labels)
        loss.backward()

