import torch
import torch.nn as nn

from ._registry import register_model

@register_model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers=[256,256], num_classes=2, dropout_rate=0.2, **kwargs):
        
        super(MLP, self).__init__()
        layers = []
        
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        
        layers.append(nn.Linear(hidden_layers[-1], num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)