import subprocess
import torch.autograd.profiler as profiler
import torch 


__all__ = ['ServerProfiler']


class ServerProfiler:
    def __init__(self):
        self.pid = None

    def start(self, use_cuda=True):
        if not use_cuda:
            print("ERROR: profiling on cpu is currently unsupported")
            return None, None

        if not torch.cuda.is_available():
            print("ERROR: use_cuda selected for profiling while cuda is not available")
            return None, None
        
        command = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv", "-l", "1"]
        self.pid = subprocess.Popen(command, stdout=subprocess.PIPE)

    def end(self):
        if not self.pid:
            print("ERROR: No process initialized, use Profiler.start_profiling to initialize a process")
        self.pid.kill()
        return self._parse_nvidia_output(self.pid.stdout.readlines())

    def _parse_nvidia_output(self, lines):
        '''
        Returns the peak memory in MiB
        '''
        lines = list(map(lambda l: int(l.decode("utf-8").strip().split(' ')[0]), lines[1:]))
        return max(lines), sum(lines)/len(lines), len(lines)



if __name__ == '__main__':
    from torch import randn, randint
    from torch.nn.functional import nll_loss
    from torchvision.models import resnet18, resnet34
    from edgify.models import resnet18 as resnet18_sparse
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    profiler = Profiler()

    profiler.start_profiling(use_cuda=True)

    model = resnet18_sparse().to(device)
    inputs = randn(64, 3, 224, 224, device=device)
    labels = randint(2, (64, ), device=device)
    loss_fn = nll_loss
    out = model(inputs)
    loss = loss_fn(out, labels)
    loss.backward()

    print(profiler.end_profiling())

    