import os
import argparse
import torch
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
    parser.add_argument("--epochs", type=int, default=25, \
        help="Number of epochs.")
    parser.add_argument("--output_dir", type=str, \
        help="Output directory for saving the trained model.")
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    print()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    dataloaders, class_names, dataset_sizes = \
                load_dataset(args.dataset_name, batch_size=args.batch_size)
    
    model = TLModel(model_name=args.model, 
                    num_classes=len(class_names), 
                    tl_strategy=args.tl_strategy)
    model.to(device)

    avg_sparsity_ratio = calc_zero_activations_percentages(model, dataloaders['val'], 
                            device=device, layer_types=[torch.nn.ReLU, torch.nn.ReLU6], 
                            verbose=False, plot=False, model_name=args.model)
    print('Average Sparsity Ratio = {:4f} %'.format(100*avg_sparsity_ratio))

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