import sys
import torch.autograd.profiler as profiler
import torch 

def profile(model, inputs, labels, loss_fn, use_cuda=False, export=False, name='model_training'):
    '''
    Returns the memory usage and compute time of one training loop (forward, backward).

    Parameters:
            model (torch.nn.Module): A model defined in PyTorch
            inputs (torch.tensor): A batch tensor input to the model
            labels (torch.tensor): Labels for the batch to calculate the loss
            loss_fn: A function used to compute the loss (and run backward() on)
            use_cuda (boolean): whether to run the training in gpu or cpu
    Returns:
            mem_usage (int): Memory usage in MB
            compute_time (float): Total time spent for training in ms
    '''
    if use_cuda and not torch.cuda.is_available():
        print("Warning: use_cuda selected for profiling while cuda is not available, \
                computations will take place on CPU instead.")
        use_cuda = False
    
    with profiler.profile(profile_memory=True, use_cuda=use_cuda) as prof:
        with profiler.record_function(name):
            out = model(inputs)
            loss = loss_fn(out, labels)
            loss.backward()

    if use_cuda:
        mem_usage = prof.total_average().cuda_memory_usage
        compute_time = prof.total_average().self_cuda_time_total
        table = prof.key_averages().table(sort_by="cuda_memory_usage")
    else:
        mem_usage = prof.total_average().cpu_memory_usage
        compute_time = prof.total_average().self_cpu_time_total
        table = prof.key_averages().table(sort_by="cpu_memory_usage")
    
    if export:
        prof.export_chrome_trace(name + '.json')
        with open(name + '.txt', 'w') as f:
            f.write(table)

    mem_usage //= 1024*1024 # Memory Usage in Mega Bytes (convert from B to MB)
    compute_time //= 1000 # Compute time in milli seconds (convert from us to ms)

    return mem_usage, compute_time

if __name__ == '__main__':
    from torch import randn, randint
    from torch.nn.functional import nll_loss
    from torchvision.models import resnet18, resnet34
    
    inputs = randn(5, 3, 224, 224)
    labels = randint(2, (5, ))
    loss_fn = nll_loss

    model = resnet18()
    mem, compute = profile(model, inputs, labels, loss_fn)
    print(mem, compute)

    model = resnet34()
    mem, compute = profile(model, inputs, labels, loss_fn)
    print(mem, compute)