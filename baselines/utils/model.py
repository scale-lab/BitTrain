import torch 
from torchvision import models
from enum import Enum

class TLStrategy(Enum):
    full_retraining = 1
    freeze_feature_extractor_all = 2
    freeze_feature_extractor_weights_only = 3


class TLModel(torch.nn.Module):
    def __init__(self, model_name, fc_hidden_dim, num_classes, 
                 tl_strategy=TLStrategy.full_retraining):
        super(TLModel, self).__init__()
            
        self.model = getattr(models, model_name)(pretrained=True)

        # Freeze layers according to chosen strategy
        if tl_strategy == TLStrategy.freeze_feature_extractor_all:
            self._freeze_all()
        elif tl_strategy == TLStrategy.freeze_feature_extractor_weights_only:
            self._freeze_weights()

        # Add fully connected layers for classification
        num_ftrs = self.model.fc.in_features

        fc_layers = []
        for hidden_dim in fc_hidden_dim:
            fc_layers.append(torch.nn.Linear(num_ftrs, hidden_dim))
            fc_layers.append(torch.nn.ReLU(inplace=True))
            num_ftrs = hidden_dim

        fc_layers.append(torch.nn.Linear(num_ftrs, num_classes))
        self.model.fc = torch.nn.Sequential(*fc_layers)

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
