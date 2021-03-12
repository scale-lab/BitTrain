import pytest
import torch
from edgify_tensor import BitmapTensor


class TestBitmapTensor:
    def test_save_load(self):
        x = torch.randn((4, 2))
        x[0][0] = 0
        x[2][1] = 0
        b = BitmapTensor(x)
        y = b.get_dense()

        assert torch.equal(x, y) == True

    