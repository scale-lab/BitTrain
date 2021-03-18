import pytest
import torch
from edgify_tensor import BitmapTensor


class TestBitmapTensor:
    def test_save_load_1(self):
        x = torch.randn((4, 2))
        x[0][0] = 0
        x[2][1] = 0
        b = BitmapTensor(x)
        y = b.get_dense(False)

        assert torch.equal(x, y) == True

    def test_save_load_2(self):
        x = torch.randn((8, 4, 2))
        x[0][0][0] = 0
        x[2][1][0] = 0
        b = BitmapTensor(x)
        y = b.get_dense(False)

        assert torch.equal(x, y) == True
    
    def test_save_load_3(self):
        x = torch.randn((16, 8, 4, 2))
        x[0][0][0][0] = 0
        x[2][1][0][1] = 0
        x[4][5][0][1] = 0
        x[6][1][3][1] = 0
        b = BitmapTensor(x)
        y = b.get_dense(False)

        assert torch.equal(x, y) == True
    
    def test_save_load_gpu(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = torch.randn((16, 8, 4, 2)).to(device)

        x[0][0][0][0] = 0
        x[2][1][0][1] = 0
        x[4][5][0][1] = 0
        x[6][1][3][1] = 0
        b = BitmapTensor(x)
        y = b.get_dense(torch.cuda.is_available())

        assert torch.equal(x, y) == True

