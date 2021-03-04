import torch
import collections
from functools import partial
import numpy as np
import matplotlib.pyplot as plt 

'''
_activations is a dictionary to save the activations,
Key - the layer name
Value - the activation
'''
_activations = collections.defaultdict(list)

'''
_sparsity_ratios is a dictionary to save the sparsity ratios of each layer,
Key - the layer name
Value - list of sparsity ratios for the executed forward passes
'''
_sparsity_ratios = collections.defaultdict(list)

def _save_activation(name, mod, inp, out):
    _activations[name].append(out.detach())

def _place_hooks(model, layers_types):
    for name, module in model.named_modules():
        if type(module) in layers_types:
            module.register_forward_hook(partial(_save_activation, name))

'''
_update_sparsity_ratios_dict: updates the sparsity ratios dictinary 
                            according to new activations in _activations
Note:: Remember to clear _activations after updating the sparsity 
    ratios dictionary to avoid accumulated activations
'''
def _update_sparsity_ratios_dict():
    for activ_layer in _activations:
        activation = torch.cat(_activations[activ_layer], dim=0)
        total_elements = torch.numel(activation)
        non_zero_elements = torch.count_nonzero(activation)
        sparsity_ratio = 1 - (non_zero_elements/total_elements)

        # Update _sparsity_ratios dictionary with new inputs
        _sparsity_ratios[activ_layer].append(sparsity_ratio.item())

def _compute_sparsity_ratios_at_hooks(model, dataloader, device):
    for inputs, _ in dataloader:
        # Perform the forward path to save activations
        inputs = inputs.to(device)
        model(inputs)

        # Update the sparsity matrix and clear the activations
        _update_sparsity_ratios_dict()
        #_activations.clear()

def calc_zero_activations_percentages(model, dataloader, device, \
                                     layer_types, verbose=False, \
                                     plot=False, model_name='Model'):
    # Place the hooks at the required layer type
    _place_hooks(model, layer_types)

    # Compute sparsity ratios
    _compute_sparsity_ratios_at_hooks(model, dataloader, device)

    # Print average sparsity ratios
    total_avg_sparsity = []
    for layer_name in _sparsity_ratios:
        avg_sparsity = np.mean(_sparsity_ratios[layer_name])
        if verbose:
            print('Layer {} - average zero activations percentage is {:.2f} %'.format(layer_name, 100*avg_sparsity))
        total_avg_sparsity.append(avg_sparsity)
    total_avg = np.mean(total_avg_sparsity)
    if verbose:
        print('All - average zero activations percentage is {:.2f} %'.format(100*total_avg))

    if plot:
        # Plot Sparsity Ratio per activation layer bar chart
        plt.bar(["Layer "+str(i) for i in range(len(_sparsity_ratios))], total_avg_sparsity, width=0.6)
        plt.xticks(rotation = 45)
        plt.ylim(top=1)
        plt.ylabel("Sparsity Ratio")
        plt.xlabel("Activation Layers")
        plt.title(f'Sparsity Ratio per activation layer for {model_name}')
        plt.tight_layout()
        plt.savefig(f'{model_name}_sparsity_per_layer.png')
    return total_avg