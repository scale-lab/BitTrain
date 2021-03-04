import subprocess
import re
import time
import torch 


__all__ = [
    'Profiler',
    ]


class Profiler:
    def __init__(self, platform='server'):
        self.pid = None
        self.platform = platform

    def start(self, use_cuda=True):
        if not use_cuda:
            print("ERROR: profiling on cpu is currently unsupported")
            return

        if not torch.cuda.is_available():
            print("ERROR: use_cuda selected for profiling while cuda is not available")
            return
        
        if self.platform == 'server':
            command = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv", "-l", "1"]
        elif self.platform == 'nano':
            command = ["tegrastats", "--interval", "1000"]
        else:
            print("ERROR: unknown platform.")
            return

        self.pid = subprocess.Popen(command, stdout=subprocess.PIPE)

    def end(self):
        if not self.pid:
            print("ERROR: No process initialized, use Profiler.start_profiling to initialize a process")
        self.pid.kill()
        if self.platform == 'server':
            return self._parse_server_output(self.pid.stdout.readlines())
        elif self.platform == 'nano':
            return self._parse_nano_output(self.pid.stdout.readlines())
        else:
            print("ERROR: unknown platform.")
            return None, None, None

    def _parse_server_output(self, lines):
        '''
        Returns the peak memory in MiB
        '''
        lines = list(map(lambda l: int(l.decode("utf-8").strip().split(' ')[0]), lines[1:]))
        return max(lines), sum(lines)/len(lines), len(lines)
    
    def _parse_nano_output(self, lines):
        '''
        Returns the peak memory in MiB
        '''
        lines = list(map(lambda l: l.decode("utf-8").strip(), lines))
        memory_usage = []
        for line in lines:
            match = re.search('RAM (?P<mem_util>[0-9]+)/[0-9]+MB' ,line)
            mem = float(match.group('mem_util'))
            memory_usage.append(mem)

        return max(memory_usage), sum(memory_usage)/len(memory_usage), len(memory_usage)


if __name__ == '__main__':
    from torch import randn, randint
    from torch.nn.functional import nll_loss
    from torchvision.models import resnet18, resnet34
    from edgify.models import resnet18 as resnet18_sparse
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    profiler = Profiler(platform='nano')
    profiler.start(use_cuda=True)
    time.sleep(5)
    mem_sys_peak, mem_sys_avg, _ = profiler.end()

    profiler.start(use_cuda=True)
    model = resnet18_sparse().to(device)
    inputs = randn(64, 3, 224, 224, device=device)
    labels = randint(2, (64, ), device=device)
    loss_fn = nll_loss
    out = model(inputs)
    loss = loss_fn(out, labels)
    loss.backward()

    mem_peak, mem_avg, prtime = profiler.end()
    mem_peak -= mem_sys_peak
    mem_avg -= mem_sys_avg
    
    print(f'Peak memory: {mem_peak:.4f} MB, Average memory: {mem_avg:.4f} MB, Time: {prtime} sec.')

    