import torch
from torch import randn, randint
from torch.nn.functional import nll_loss
from torchvision.models import resnet18, resnet34, mobilenet_v2
import matplotlib.pyplot as plt
from edgify.profiling.functions import profile

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = mobilenet_v2().to(device)
    inputs = randn(16, 3, 224, 224, device=device)
    labels = randint(2, (16, ), device=device)
    loss_fn = nll_loss
    mem, compute = profile(model, inputs, labels, loss_fn, use_cuda=torch.cuda.is_available(), export=True)

    print(f'Mobilenet - Memory Footprint: {mem} MB - Latency: {compute} ms')

