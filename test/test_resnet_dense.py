import pytest
import torch

from edgify import models
from torch.nn.functional import nll_loss

class TestResnetDense:
    def test_forward(self):
        model = models.resnet18(bitmap=False)
        data = torch.rand((8, 3, 224, 224))

        out = model(data)

    def test_backward(self):
        model = models.resnet101(bitmap=False)
        inputs = torch.rand((8, 3, 224, 224))
        labels = torch.randint(2, (8, ))

        model.eval()
        out = model(inputs)
        loss = nll_loss(out, labels)
        loss.backward()

