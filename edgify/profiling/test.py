from torch import randn, randint
from torch.nn.functional import nll_loss
from torchvision.models import resnet18, resnet34, mobilenet_v2
import matplotlib.pyplot as plt
from functions import profile

def compare_batchsize_cpu_vs_cuda(verbose=False):
    models = [resnet18(), resnet34(), mobilenet_v2()]
    names = ["resnet18", "resnet34", "mobilenet"]
    colors = ['r', 'g', 'b']
    for i, (model, name, color) in enumerate(zip(models, names, colors)):
        cpu_mem = []
        cuda_mem = []

        cpu_compute = []
        cuda_compute = []
        
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        for batch_size in batch_sizes:
            if verbose:
                print(f'Batch Size {batch_size}')

            inputs = randn(batch_size, 3, 224, 224)
            labels = randint(2, (batch_size, ))
            loss_fn = nll_loss

            # Profile execution on CPU
            mem, compute = profile(model, inputs, labels, loss_fn, use_cuda=False, name=str(name+"CPU"))
            cpu_mem.append(mem)
            cpu_compute.append(compute)
            if verbose:
                print(f'Model {name} - CPU - Memory Footprint: {mem} MB - Latency: {compute} ms')

            # Profile execution on CUDA
            mem, compute = profile(model, inputs, labels, loss_fn, use_cuda=True, name=str(name+"CUDA"))
            cuda_mem.append(mem)
            cuda_compute.append(compute)
            if verbose:
                print(f'Model {name} - CUDA - Memory Footprint: {mem} MB - Latency: {compute} ms')
            
        plt.figure(0)
        plt.plot(batch_sizes, cpu_compute, f'{color}-', label=f"CPU {name}")
        plt.plot(batch_sizes, cuda_compute, f'{color}--', label=f"CUDA {name}")

        plt.xlabel("Batch Size")
        plt.ylabel("Latency (ms)")
        plt.grid()
        plt.legend()
        plt.title(f"Latency (ms) over changing batch size")
        plt.savefig("compute_batch_size.png")

        plt.figure(1)
        plt.plot(batch_sizes, cpu_compute, f'{color}-', label=f"CPU {name}")
        plt.plot(batch_sizes, cuda_compute, f'{color}--', label=f"CUDA {name}")

        plt.xlabel("Batch Size")
        plt.ylabel("Memory (MB)")
        plt.grid()
        plt.legend()
        plt.title(f"Memory (MB) over changing batch size")
        plt.savefig("memory_batch_size.png")

def compare_itterations_cpu_vs_cuda(verbose=False):
    models = [resnet18(), resnet34(), mobilenet_v2()]
    names = ["resnet18", "resnet34", "mobilenet"]
    colors = ['r', 'g', 'b']
    for i, (model, name, color) in enumerate(zip(models, names, colors)):
        cpu_mem = []
        cuda_mem = []

        cpu_compute = []
        cuda_compute = []
        
        batch_size = 8
        itterations = [1, 10, 20, 50, 100]
        for itter in itterations:
            if verbose:
                print(f'Batch Size {batch_size} - Itterations {itter}')

            inputs = randn(batch_size, 3, 224, 224)
            labels = randint(2, (batch_size, ))
            loss_fn = nll_loss

            # Profile execution on CPU
            mem, compute = profile(model, inputs, labels, loss_fn, use_cuda=False, name=str(name+"CPU"), itterations=itter)
            cpu_mem.append(mem)
            cpu_compute.append(compute)
            if verbose:
                print(f'Model {name} - CPU - Memory Footprint: {mem} MB - Latency: {compute} ms')

            # Profile execution on CUDA
            mem, compute = profile(model, inputs, labels, loss_fn, use_cuda=True, name=str(name+"CUDA"), itterations=itter)
            cuda_mem.append(mem)
            cuda_compute.append(compute)
            if verbose:
                print(f'Model {name} - CUDA - Memory Footprint: {mem} MB - Latency: {compute} ms')
            
        plt.figure(0)
        plt.plot(itterations, cpu_compute, f'{color}-', label=f"CPU {name}")
        plt.plot(itterations, cuda_compute, f'{color}--', label=f"CUDA {name}")

        plt.xlabel("Itterations")
        plt.ylabel("Latency (ms)")
        plt.grid()
        plt.legend()
        plt.title(f"Average Latency (ms) over changing itterations")
        plt.savefig("compute_itterations.png")

        plt.figure(1)
        plt.plot(itterations, cpu_compute, f'{color}-', label=f"CPU {name}")
        plt.plot(itterations, cuda_compute, f'{color}--', label=f"CUDA {name}")

        plt.xlabel("Itterations")
        plt.ylabel("Memory (MB)")
        plt.grid()
        plt.legend()
        plt.title(f"Average Memory (MB) over changing itterations")
        plt.savefig("memory_itterations.png")
    
if __name__ == '__main__':
    compare_batchsize_cpu_vs_cuda(verbose=True)
    compare_itterations_cpu_vs_cuda(verbose=True)