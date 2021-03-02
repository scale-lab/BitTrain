import torch

class Linear(torch.auto.autograd):

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input.to_sparse(), weight.to_sparse(), bias.to_sparse() if bias else None)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        input, weight, bias = input.to_dense(), weight.to_dense(), bias.to_dense() if bias else None
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        
        return grad_input, grad_weight, grad_bias

