import os
import argparse
import torch
from utils.dataset import load_dataset
from utils.train import train_model
from utils.model import TLModel, TLStrategy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--model", type=str, default="resnet18", \
        help="Name of the model you want from PyTorch model zoo.")
    parser.add_argument("--dataset_name", type=str, default="cifar10", \
        help="Name of the dataset you want from PyTorch dataset zoo.")
    parser.add_argument("--fc_hidden_dim", nargs="+", default=[1024,256], \
        help="List of the hidden layers dimensions for the classifier.")
    parser.add_argument("--batch_size", type=int, default=16, \
        help="Batch size.")
    parser.add_argument("--epochs", type=int, default=25, \
        help="Number of epochs.")
    parser.add_argument("--output_dir", type=str, \
        help="Output directory for saving the trained model.")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    dataloaders, class_names, dataset_sizes = \
                load_dataset(args.dataset_name, batch_size=args.batch_size)
    
    model = TLModel(model_name=args.model, 
                    fc_hidden_dim=args.fc_hidden_dim,
                    num_classes=len(class_names), 
                    tl_strategy=TLStrategy.freeze_feature_extractor_weights_only)
    model.to(device)

    # Re-train the last fully connected layers and biases
    model = train_model(model, dataloaders, dataset_sizes, device=device, num_epochs=args.epochs)
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        file_name = os.path.join(args.output_dir, "model.pt")
        torch.save(model.state_dict(), file_name)