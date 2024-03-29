import torch
from torch.nn.functional import conv2d

from edgify_tensor import BitmapTensor

class Conv2d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
        ctx.stride, ctx.padding, ctx.dilation, ctx.groups = stride, padding, dilation, groups
        ctx.input = BitmapTensor(input)
        ctx.save_for_backward(weights, bias if bias else None)
        with torch.no_grad():
            result = conv2d(input, weights, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=1)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        stride, padding, dilation, groups = ctx.stride, ctx.padding, ctx.dilation, ctx.groups
        input = ctx.input.get_dense(grad_output.device != torch.device("cpu"))
        weights, bias = ctx.saved_tensors
        
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weights, grad_output, 
                                                    stride=stride, padding=padding,
                                                    dilation=dilation, groups=groups)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weights.shape, grad_output,
                                                      stride=stride, padding=padding,
                                                      dilation=dilation, groups=groups)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        
        return grad_input, grad_weight, grad_bias, None, None, None, None

    