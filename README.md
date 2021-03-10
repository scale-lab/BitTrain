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
We perform baselines on a transfer learning task. Refer to the [Pytorch Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html). We support three transfer learning strategies:
1. Full retraining. 
2. Freeze feature extractor retraining.
3. Freeze feature extractor weights only retraining.

To run the baselines:

```Shell
cd baselines
python run_baseline.py --model {MODEL_NAME}
                       --dataset_name {DATASET_NAME}
                       --tl_strategy {TL_STRATEGY}
```
- `MODEL_NAME` is the name of the model from [PyTorch model zoo](https://pytorch.org/vision/0.8/models.html).
- `DATASET_NAME` is the name of the dataset from [PyTorch dataset zoo](https://pytorch.org/vision/0.8/datasets.html).
- `TL_STRATEGY` is the transfer learning strategy number as mentioned above.

## Sparse Tensor

Make sure you have the latest version of `clang` compiler. Then, build our sparse tensor using: `make install`.

Use it:

```Python
import torch
from edgify_tensor import BitmapTensor  # still WIP


```

## License
BSD-3

