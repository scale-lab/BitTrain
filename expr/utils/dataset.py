import os
import torch
from torchvision import datasets, transforms


def load_dataset(dataset_name, batch_size=4):
    if dataset_name == "cifar10":
        return _load_cifar10(batch_size)
    else:
        print("ERROR:: Dataset {dataset_name} is not available.")

def _load_cifar10(batch_size=4):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root='../../data/', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='../../data/', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    class_names = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    dataloaders = {'train':trainloader, 'val':testloader}
    dataset_sizes = {'train':len(trainset), 'val':len(testset)}

    return dataloaders, class_names, dataset_sizes