import torch
import torchvision
import collections
import numpy as np

'''
_sparsity_ratios is a dictionary to save the sparsity ratios of each layer,
Key - the layer name
Value - list of sparsity ratios for the executed forward passes
'''
_sparsity_ratios_per_layer = collections.defaultdict(list)
_sparsity_ratios_per_layer_type = collections.defaultdict(list)
_total_memory_per_layer_type = collections.defaultdict(list)
_bitmap_memory_footprint = collections.defaultdict(list)
_dense_memory_footprint = collections.defaultdict(list)
_activations_stats_for_hist = []

_layers_types = [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Dropout,
                torch.nn.Linear, torch.nn.MaxPool2d, torch.nn.AdaptiveAvgPool2d]

_layers_names = {torch.nn.Conv2d:"Conv", 
                torch.nn.BatchNorm2d: "BatchNorm", 
                torch.nn.Dropout: "Dropout",
                torch.nn.Linear: "Linear",
                torch.nn.MaxPool2d: "MaxPool",
                torch.nn.AdaptiveAvgPool2d: "AvgPool"}

class Hook():
    def __init__(self, module, name, pre=False):
        self.name = name
        self.type = _layers_names[type(module)]
        if pre==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_forward_pre_hook(self.hook_pre_fn)
    
    def hook_fn(self, module, input, output):
        assert len(input) == 1
        self.input = input[0].detach()
        self.output = output.detach()

    def hook_pre_fn(self, module, input):
        assert len(input) == 1
        self.input = input[0].detach()

    def close(self):
        self.hook.remove()

'''
_place_hooks: places hooks at the given layer types
'''
def _place_hooks(model, layers_types):
    hooks = []
    for name, module in model.named_modules():
        if type(module) in layers_types:
            hooks.append(Hook(module, name, pre=True))
        else:
            print("Skipped", name, type(module))
    return hooks

'''
_update_sparsity_ratios_dict: updates the sparsity ratios dictinary 
                            according to new values at hooks
'''
def _update_sparsity_ratios_dict(hooks):
    for hook in hooks:
        activation = hook.input
        _activations_stats_for_hist.extend(list(activation.view(-1)))
        total_elements = torch.numel(activation)
        non_zero_elements = torch.count_nonzero(torch.abs(activation) > 0.001)
        sparsity_ratio = 1 - (non_zero_elements/total_elements)
        _sparsity_ratios_per_layer[hook.name].append(sparsity_ratio.item())
        _sparsity_ratios_per_layer_type[hook.type].append(sparsity_ratio.item())
        _total_memory_per_layer_type[hook.type].append(total_elements)
        _bitmap_memory_footprint[hook.name].append(total_elements*1 + non_zero_elements*4)
        _dense_memory_footprint[hook.name].append(np.prod(list(activation.shape))*4)

'''
_compute_sparsity_ratios_at_hooks: loop on dataset and 
                                calculate the sparsity at each layer
'''
def _compute_sparsity_ratios_at_hooks(model, hooks, dataloader, device):
    for inputs, _ in dataloader:
        # Perform the forward path to save activations
        print(inputs.shape)
        inputs = inputs.to(device)
        model(inputs)

        # Update the sparsity matrix and clear the activations
        _update_sparsity_ratios_dict(hooks)
        break
'''
_replace_relu_inplace_to_relu: used as a workaround because hooks work wrong 
                        with inplace operations, replace each inplace ReLU 
                        with similar one with implace = False
'''
def _replace_relu_inplace_to_relu(model, relu_type):
    for child_name, child in model.named_children():
        if isinstance(child, relu_type):
            setattr(model, child_name, relu_type(inplace=False))
        else:
            _replace_relu_inplace_to_relu(child, relu_type)

def calc_zero_activations_percentages(model, dataloader, \
                                     name, device, verbose=False):

    # Workaround:: Replace the RELU inplace to normal because 
    # the hooks work wrong with ReLU inplace
    relu_types = [torch.nn.ReLU6, torch.nn.ReLU]

    for layer in relu_types:
        _replace_relu_inplace_to_relu(model, layer)
    
    print(model)
    # Place the hooks at the required layer type
    hooks = _place_hooks(model, _layers_types)

    # Compute sparsity ratios
    _compute_sparsity_ratios_at_hooks(model, hooks, dataloader, device)

    # Reemove hooks
    for hook in hooks:
        hook.close()
    
    # Print average sparsity ratios
    avg_sparsity_per_layer = []
    avg_saving_to_dense_per_layer = []
    
    for layer_name in _sparsity_ratios_per_layer:
        avg_sparsity = np.mean(_sparsity_ratios_per_layer[layer_name])
        avg_saving_to_dense = 1 - np.mean(_bitmap_memory_footprint[layer_name])/ \
                                  np.mean(_dense_memory_footprint[layer_name])
        
        if avg_sparsity > 0.15:
            avg_saving_to_dense_per_layer.append(avg_saving_to_dense)
            avg_sparsity_per_layer.append(100*avg_sparsity)
        else:
            avg_saving_to_dense_per_layer.append(0)
            avg_sparsity_per_layer.append(0)

        if verbose:
            print('Layer {} - input sparsity is {:.2f} %, saved {:.2f}% than dense \
                        and {:.2f}% than COO'.format(layer_name, 100*avg_sparsity, \
                        100*avg_saving_to_dense))
        
    total_avg = np.mean(avg_sparsity_per_layer)
    if verbose:
        print('All - average zero activations percentage is {:.2f} %'.format(total_avg))
        print("Average Saving compared to dense is {:.2f}".format(100*np.mean(avg_saving_to_dense_per_layer)))
        
    avg_sparsity_per_layer_type = []
    total_memory = []
    layer_types = []
    for layer_type in _sparsity_ratios_per_layer_type:
        avg_sparsity = np.mean(_sparsity_ratios_per_layer_type[layer_type])
        if verbose:
            print('Layer {} - input sparsity is {:.4f} %'.format(layer_type, 100*avg_sparsity))
        avg_sparsity_per_layer_type.append(100*avg_sparsity)
        layer_types.append(layer_type)
        total_memory.append(np.sum(_total_memory_per_layer_type[layer_type]))
    
    total_memory_percentage = []
    for idx, value in enumerate(total_memory):
        total_memory_percentage.append(value/np.sum(total_memory))

    return avg_sparsity_per_layer, avg_sparsity_per_layer_type, total_memory_percentage, \
             layer_types, _activations_stats_for_hist