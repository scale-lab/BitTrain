import os
import sys
import argparse
import torch
import copy 
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import random

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from utils.dataset import load_dataset
from utils.train import train_model
from utils.model import TLModel
from utils.sparsity_stats import calc_zero_activations_percentages

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--model", type=str, default="resnet18", \
        help="Name of the model you want from PyTorch model zoo.")
    parser.add_argument("--dataset_name", type=str, default="cifar10", \
        help="Name of the dataset you want from PyTorch dataset zoo.")
    parser.add_argument("--batch_size", type=int, default=8, \
        help="Batch size.")
    parser.add_argument("--tl_strategy", type=int, default=1, \
        help="Transfer learning strategy: 1 for full_retraining, \
                2 for freeze_feature_extractor_all, \
                3 for freeze_feature_extractor_weights_only.")
    parser.add_argument("--epochs", type=int, default=2, \
        help="Number of epochs.")
    parser.add_argument("--output_dir", type=str, \
        help="Output directory for saving the trained model.")
    parser.add_argument("--load", help="Load pretrained model", default="", type=str)
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    print()

    device = "cpu"
    
    dataloaders, num_classes, dataset_sizes = \
                load_dataset(args.dataset_name, batch_size=args.batch_size)

    model = TLModel(model_name=args.model, 
                    num_classes=num_classes, 
                    tl_strategy=args.tl_strategy)
            
    model.to(device)

    if args.load:
        model.load_state_dict(torch.load(args.load))
    
    avg_sparsity_per_layer, avg_sparsity_per_layer_type, total_memory, layer_types, _activations = calc_zero_activations_percentages(model, dataloaders['val'], 
                            name=args.model, device=device, 
                            verbose=True)
    print('Average Sparsity Ratio = {:4f} %'.format(np.mean(avg_sparsity_per_layer)))

    # Plot Sparsity Ratio per activation layer bar chart
    plt.figure(0)
    fig, ax = plt.subplots()
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    plt.bar([layer_types[i] for i in range(len(avg_sparsity_per_layer_type))], 
                    avg_sparsity_per_layer_type, width=0.6, color="royalblue", edgecolor = "black")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel("Activations Sparsity Ratio (%)", fontsize=14)
    plt.xlabel("Layer Type", fontsize=14)
    # plt.title(f'Sparsity Ratio per layer type for {args.model}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{args.model}_sparsity_per_layer_type.png')

    plt.figure(1)
    fig, ax = plt.subplots()
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    plt.hist(avg_sparsity_per_layer, weights=np.ones(len(avg_sparsity_per_layer)) / len(avg_sparsity_per_layer),
                         bins=10, facecolor='royalblue')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    plt.ylabel("Percentage (%)", fontsize=14)
    plt.xlabel("Activations Sparsity Ratio", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{args.model}_sparsity_hist_per_layer.png')

    plt.figure(2)
    fig, ax = plt.subplots()
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    plt.bar([layer_types[i] for i in range(len(total_memory))], 
                    total_memory, width=0.6, color="royalblue", edgecolor = "black")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel("Memory Footprint (%)", fontsize=14)
    plt.xlabel("Layer Type", fontsize=14)
    # plt.title(f'Sparsity Ratio per layer type for {args.model}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{args.model}_memory_per_layer_type.png')

    print(len(_activations))
    #_activations_sub = random.sample(_activations, 10000)
    # _activations_sub = _activations
    # _activations_sub = np.asarray(_activations_sub)
    # _activations_sub = np.abs(np.clip(_activations_sub, -2, 2))
    # plt.figure(3)
    # fig, ax = plt.subplots()
    # ax.set_axisbelow(True)
    # ax.yaxis.grid(color='gray', linestyle='dashed')
    # plt.hist(_activations_sub, weights=np.ones(len(_activations_sub)) / len(_activations_sub),
    #                      bins=20, facecolor='royalblue')
    # plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    # plt.ylabel("#", fontsize=14)
    # plt.xlabel("Activations Values", fontsize=14)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.tight_layout()
    # plt.savefig(f'{args.model}_activations_hist_per_layer.png')