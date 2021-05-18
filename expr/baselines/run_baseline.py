import os
import sys
import argparse
import torch

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
    parser.add_argument("--batch_size", type=int, default=16, \
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    dataloaders, num_classes, dataset_sizes = \
                load_dataset(args.dataset_name, batch_size=args.batch_size)
    
    model = TLModel(model_name=args.model, 
                    num_classes=num_classes, 
                    tl_strategy=args.tl_strategy)
    model.to(device)

    if args.load:
        model.load_state_dict(torch.load(args.load))

    # Re-train the complete model
    model, accuracy, time = train_model(model, dataloaders, dataset_sizes, device=device, num_epochs=args.epochs)

    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        file_name = os.path.join(args.output_dir, "model.pt")
        torch.save(model.state_dict(), file_name)

        log_file_name = os.path.join(args.output_dir, "training.log")
        with open(log_file_name, 'w') as f:
            for arg in vars(args):
                f.write(f'{arg}: {getattr(args, arg)} \n')
            f.write(f'Accuracy: {accuracy*100:.4f} % \n')
            f.write(f'Training Time: {time:.4f} % \n')

