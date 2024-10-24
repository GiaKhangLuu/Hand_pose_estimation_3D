import torch
import torch.nn as nn
import torch.optim as optim

class ANN(nn.Module):
    def __init__(self, input_dim=322, output_dim=144, hidden_dim=256, num_hidden_layers=2, dropout_rate=0.4):
        super(ANN, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim, affine=False))
        layers.append(nn.SiLU())

        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim, affine=False))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.BatchNorm1d(output_dim, affine=False))
        layers.append(nn.SiLU())
        layers.append(nn.Dropout(dropout_rate))
        
        layers.append(nn.Linear(output_dim, output_dim))
        layers.append(nn.BatchNorm1d(output_dim, affine=False))
        layers.append(nn.SiLU())
        layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(output_dim, output_dim))

        self.network = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.constant_(layer.bias, 0.5)

    def forward(self, x):
        return self.network(x)
