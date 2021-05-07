import torch
import collections
from torch import randn, randint
from torch.nn.functional import nll_loss
from torchvision.models import resnet18, resnet34, resnet50, mobilenet_v2, alexnet, squeezenet1_0 
import matplotlib.pyplot as plt
from functools import partial
import numpy as np

_activations = collections.defaultdict(list)

def _to_float_MB(value):
    # float is 4 bytes
    return (value*4)/(1024*1024)

def _count_parameters(model):
    num_parameters = 0
    for param in model.parameters():
        num_parameters += torch.numel(param)
    return _to_float_MB(num_parameters)

def _count_gradients(model):
    num_gradients = 0
    for param in model.parameters():
        num_gradients += torch.numel(param.grad)
    return _to_float_MB(num_gradients)

def _save_activation(name, mod, inp, out):
    _activations[name].append(out.detach())

def _place_hooks(model):
    for name, module in model.named_modules():
        if type(module) in [torch.nn.ReLU, torch.nn.ReLU6]: 
            # print("Skip", type(module))
            continue
        module.register_forward_hook(partial(_save_activation, name))

def _count_activations(model, inputs):
    _place_hooks(model)
    model(inputs)

    num_activations = 0
    for activ_layer in _activations:
        activations = _activations[activ_layer]
        num_activations += sum([torch.numel(activation) for activation in activations])
    
    return _to_float_MB(num_activations)

def compare_params_activ_different_models(batch_sz=8):
    models = [mobilenet_v2(), resnet34(), resnet50()]
    names = ["Mobilenet-v2", "ResNet-34", "ResNet-50"]
    
    params = []
    grads = []
    activations = []

    for model, name in zip(models, names):
        _activations.clear()

        inputs = randn(batch_sz, 3, 224, 224)
        labels = randint(2, (batch_sz, ))
        loss_fn = nll_loss

        activations.append(_count_activations(model, inputs))

        out = model(inputs)
        loss = loss_fn(out, labels)
        loss.backward()

        grads.append(_count_gradients(model))
        params.append(_count_parameters(model))

    params = np.asarray(params)
    grads = np.asarray(grads)
    activations = np.asarray(activations)

    fig, ax = plt.subplots()
    ax.set_axisbelow(True)
    ax.xaxis.grid(color='gray', linestyle='dashed')
    ind = np.arange(len(models))
    width = 0.6
    p1 = ax.barh(ind, activations, width, label='Activations', edgecolor = "black", color="royalblue")
    p2 = ax.barh(ind, params, width, left=activations, label='Parameters', edgecolor = "black", color="darkorange")
    p3 = ax.barh(ind, grads, width, left=params+activations, label='Gradients', edgecolor = "black", color="green")

    ax.set_xlabel('Memory Footprint (MB)', fontsize=14)
    ax.set_yticks(ind)
    ax.set_yticklabels(names, fontsize=14)
    ax.invert_yaxis()
    ax.legend()
    
    fig.tight_layout()

    plt.savefig("param_activ_grad.png")

def compare_params_activ_different_batch_sizes():
    params = []
    grads = []
    activations = []
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    for batch_sz in batch_sizes:
        _activations.clear()

        model = resnet50()
        inputs = randn(batch_sz, 3, 224, 224)
        labels = randint(2, (batch_sz, ))
        loss_fn = nll_loss

        activations.append(_count_activations(model, inputs))
        out = model(inputs)
        loss = loss_fn(out, labels)
        loss.backward()

        grads.append(_count_gradients(model))
        params.append(_count_parameters(model))

    params = np.asarray(params)
    grads = np.asarray(grads)
    activations = np.asarray(activations)

    fig, ax = plt.subplots()
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ind = np.arange(len(batch_sizes))
    width = 0.6
    p2 = ax.bar(ind, params, width, label='Parameters', edgecolor = "black", color="darkorange")
    p3 = ax.bar(ind, grads, width, bottom=params, label='Gradients', edgecolor = "black", color="green")
    p1 = ax.bar(ind, activations, width, bottom=grads+params, label='Activations', edgecolor = "black", color="royalblue")

    ax.set_xlabel('Batch Size', fontsize=14)
    ax.set_ylabel('Memory Footprint (MB)', fontsize=14)
    ax.set_xticks(ind)
    ax.set_xticklabels(batch_sizes, fontsize=14)
    ax.legend()
    
    fig.tight_layout()

    plt.savefig("param_activ_grad_batch_sz.png")

if __name__ == '__main__':
    compare_params_activ_different_models(batch_sz=16)
    compare_params_activ_different_batch_sizes()