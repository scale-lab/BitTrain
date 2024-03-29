import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from edgify.functional.bitmap import Conv2d
# from edgify.functional.dense import Conv2d


__all__ = ['BitmapConv2d']


class BitmapConv2d(nn.Conv2d):
    # only override the forward function
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != 'zeros':
            return Conv2d.apply(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            self.weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return Conv2d.apply(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

