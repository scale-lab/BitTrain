import os
import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
    
def load_dataset(dataset_name, batch_size=4):
    if dataset_name == "cifar10":
        return _load_cifar10(batch_size)
    elif dataset_name == "imagenet":
        return _load_imagenet(batch_size)
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

    num_classes = 10
    
    dataloaders = {'train':trainloader, 'val':testloader}
    dataset_sizes = {'train':len(trainset), 'val':len(testset)}

    return dataloaders, num_classes, dataset_sizes

def _load_imagenet(batch_size=4, num_classes=1000):
    class SubDatasetFolder(datasets.DatasetFolder):
        def _find_classes(self, dir):
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
            classes.sort()
            classes = classes[:num_classes]  # overwritten from original to take subset
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
            return classes, class_to_idx

    input_size = 224

    traindir = os.path.join('../../data/imagenet-mini', "train/")
    valdir = os.path.join('../../data/imagenet-mini', "val/")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # Reimplements ImageFolder arguments
    trainset = SubDatasetFolder(
        traindir,
        default_loader,
        IMG_EXTENSIONS,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
    testset = SubDatasetFolder(
        valdir,
        default_loader,
        IMG_EXTENSIONS,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    dataloaders = {'train':trainloader, 'val':testloader}
    dataset_sizes = {'train':len(trainset), 'val':len(testset)}

    return dataloaders, num_classes, dataset_sizes