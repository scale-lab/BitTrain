import torch

class ReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input.to_sparse())
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        input = input.to_dense()
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


if __name__ == '__main__':
    N, D_in, H, D_out = 64, 1000, 100, 10

    x = torch.randn(N, D_in, dtype=torch.float32)
    y = torch.randn(N, D_out, dtype=torch.float32)

    w1 = torch.randn(D_in, H, dtype=torch.float32, requires_grad=True)
    w2 = torch.randn(H, D_out, dtype=torch.float32, requires_grad=True)

    learning_rate = 1e-6
    for t in range(500):
        relu = ReLU.apply

        y_pred = relu(x.mm(w1)).mm(w2)
        loss = (y_pred - y).pow(2).sum()
        
        if t % 100 == 0:
            print(t, loss.item())

        loss.backward()

        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad

            w1.grad.zero_()
            w2.grad.zero_()

    # dummy test
    x = torch.randn(3, 3, requires_grad=True)
    out = relu(x)
    print(out)
    print(out.to_sparse())
    print(out.to_sparse().to_dense())

    