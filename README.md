# Edge Training
Trying to make training on the edge memory efficient


## Setup

1. `virtualenv .venv --python=python3.7`
2. `source .venv/bin/activate`
3. `python setup.py install`


## Profiling a model

In order to measure the memory footprint for a model during training, we provide the following function:

```Python
from edgify.profiling.functions import profile

```

Example

```Python
import torch
from torch.nn.functional import nll_loss
from torchvision.models import resnet18, resnet34

inputs = torch.randn(5, 3, 224, 224)
labels = torch.randint(2, (5, ))
loss_fn = nll_loss

model = resnet18()
mem, compute = profile(model, inputs, labels, loss_fn)
print(mem, compute)     # prints memory (in KB) and total time for one training loop (forward + backward) in µs.
```

The function takes optional arguments:
- `use_cuda=True`: profile memory and time for GPU
- `export=True`: exports the results of the profiling in table and trace formats. It uses the `name` argument as the file path.


## Baselines
We perform baselines on a transfer learning task. Refer to the [Pytorch Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html). Download the dataset from [this link](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) and extract it to the `./data` folder.

### Full Re-training

```Shell
cd baselines
python full_retraining.pyt resnet18
```

You can try with different models from [PyTorch models library](https://pytorch.org/vision/0.8/models.html).

### Freeze Feature Extractor


### Bias-only Training



## License
BSD-3

