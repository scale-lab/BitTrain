import torch 
from torchvision import models
from enum import Enum

class TLStrategy(Enum):
    full_retraining = 1
    freeze_feature_extractor_all = 2
    freeze_feature_extractor_weights_only = 3

class TLModel(torch.nn.Module):
    def __init__(self, model_name, num_classes, 
                 tl_strategy=TLStrategy.full_retraining):
        super(TLModel, self).__init__()
            
        self.model = getattr(models, model_name)(pretrained=True)
        self.num_classes = num_classes

        # Freeze layers according to chosen strategy
        if tl_strategy == TLStrategy.freeze_feature_extractor_all:
            self._freeze_all()
        elif tl_strategy == TLStrategy.freeze_feature_extractor_weights_only:
            self._freeze_weights()

        # Add fully connected layers for classification
        if "resnet" in model_name:
            self.model.fc = self._get_resnet_classifier()
        elif "vgg" in model_name:
            self.model.classifier = self._get_vgg_classifier()
        elif "mobilenet" in model_name:
            self.model.classifier = self._get_mobilenet_classifier()
        elif "alexnet" in model_name:
            self.model.classifier = self._get_alexnet_classifier()
        else:
            print("Error: Model not supported!")

    def forward(self, x):
        return self.model(x)
    
    def _freeze_all(self):
        '''
        Freeze weights and biases of all layers
        '''
        for param in self.model.parameters():
            param.requires_grad = False

    def _freeze_weights(self):
        '''
        Freeze weights only of all layers
        '''
        for name, param in self.model.named_parameters():
            if "bias" not in name:
                param.requires_grad = False

    def _get_mobilenet_classifier(self):
        num_ftrs = self.model.classifier[1].in_features

        return torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=False),
            torch.nn.Linear(in_features=1280, out_features=1000, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.2, inplace=False),
            torch.nn.Linear(in_features=1000, out_features=256, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.2, inplace=False),
            torch.nn.Linear(in_features=256, out_features=self.num_classes, bias=True)
            )

    def _get_vgg_classifier(self):
        num_ftrs = self.model.classifier[0].in_features

        return torch.nn.Sequential(
            torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5, inplace=False),
            torch.nn.Linear(in_features=4096, out_features=1024, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5, inplace=False),
            torch.nn.Linear(in_features=1024, out_features=256, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5, inplace=False),
            torch.nn.Linear(in_features=256, out_features=self.num_classes, bias=True)
            )

    def _get_resnet_classifier(self):
        num_ftrs = self.model.fc.in_features

        fc_layers = []
        fc_hidden_dim = [1024, 256]
        for hidden_dim in fc_hidden_dim:
            fc_layers.append(torch.nn.Linear(num_ftrs, hidden_dim))
            fc_layers.append(torch.nn.ReLU(inplace=True))
            num_ftrs = hidden_dim

        fc_layers.append(torch.nn.Linear(num_ftrs, self.num_classes))
        return torch.nn.Sequential(*fc_layers)

    def _get_alexnet_classifier(self):
        num_ftrs = self.model.classifier[1].in_features

        return torch.nn.Sequential(
            torch.nn.Dropout(p=0.5, inplace=False),
            torch.nn.Linear(in_features=9216, out_features=4096, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5, inplace=False),
            torch.nn.Linear(in_features=4096, out_features=1024, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5, inplace=False),
            torch.nn.Linear(in_features=1024, out_features=256, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=256, out_features=self.num_classes, bias=True)
            )