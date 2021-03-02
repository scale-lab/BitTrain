import subprocess
import torch.autograd.profiler as profiler
import torch 


def _memory_peak_from_nvidia_output(lines):
    '''
    Returns the peak memory in MiB
    '''
    lines = list(map(lambda l: int(l.decode("utf-8").strip().split(' ')[0]), lines[1:]))
    return max(lines)


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
    '''
    if not use_cuda:
        print("ERROR: profiling on cpu is currently unsupported")
        return None, None

    if not torch.cuda.is_available():
        print("ERROR: use_cuda selected for profiling while cuda is not available")
        return None, None
    
    command = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv", "-l", "1"]
    pid = subprocess.Popen(command, stdout=subprocess.PIPE)
    
    out = model(inputs)
    loss = loss_fn(out, labels)
    loss.backward()
    
    pid.kill()
    return _memory_peak_from_nvidia_output(pid.stdout.readlines())


if __name__ == '__main__':
    from torch import randn, randint
    from torch.nn.functional import nll_loss
    from torchvision.models import resnet18, resnet34
    
    inputs = randn(5, 3, 224, 224)
    labels = randint(2, (5, ))
    loss_fn = nll_loss

    model = resnet18()
    mem = profile(model, inputs, labels, loss_fn)
    print(mem)

    model = resnet34()
    mem = profile(model, inputs, labels, loss_fn)
    print(mem)

