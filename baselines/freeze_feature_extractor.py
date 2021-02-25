import os
import time
import copy
import argparse
import torch
from torchvision import datasets, models, transforms
from utils.datasets import load_data
from utils.train import train_model
from edgify.profiling.functions import profile

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("model", type=str, \
        help="Name of the model you want from PyTorch model zoo.")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloaders, class_names, dataset_sizes = load_data()
    model = getattr(models, args.model)(pretrained=True)
    
    # Freeze all the convolutional layers of the network
    for param in model.parameters():
        param.requires_grad = False

    # Add fully connected layers for classification
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(class_names))

    model.to(device)

    # Do memory profiling first
    inputs, labels = next(iter(dataloaders['val']))

    inputs = inputs.to(device)
    labels = labels.to(device)

    mem_usage, compute_time = profile(model, inputs, labels, 
                                      torch.nn.CrossEntropyLoss(), 
                                      use_cuda=torch.cuda.is_available(),
                                      export=False)
    compute_time /= 1000.0

    print(f'Memory usage (Mb): {mem_usage:,}, Compute Time (sec.): {compute_time:.4f} on {device}')
    
    # Re-train the last fully connected layers
    train_model(model, dataloaders, dataset_sizes, device=device, num_epochs=25)