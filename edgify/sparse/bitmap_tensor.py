import torch


__all__ = ['BitmapTensor']


class BitmapTensor:
    '''
    This is a temporary implementation and should be moved later to C++
    '''
    @staticmethod
    def to_bitmap(t: torch.Tensor):
        bitmap_data, bitmap = None, None
        
        # extract indices of non-zero elements

        # build bitmap

        # compress elements to non-zero vector

        return bitmap_data, bitmap

    @staticmethod
    def to_tensor(bitmap_data, bitmap):
        t = torch.Tensor()

        # reconstruct from bitmap data

        return t

