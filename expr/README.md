# Baselines
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