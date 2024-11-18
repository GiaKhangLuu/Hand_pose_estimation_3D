import torch
import torch.nn as nn
import torch.optim as optim

class ANN(nn.Module):
    def __init__(
        self, 
        input_dim=322, 
        output_dim=144, 
        hidden_dim_container=[100, 100, 100], 
        num_hidden_layers=2, 
        dropout_rate=0.4
    ):

        assert len(hidden_dim_container) == num_hidden_layers + 1

        super().__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim_container[0]))
        layers.append(nn.BatchNorm1d(hidden_dim_container[0], affine=False))
        layers.append(nn.SiLU())

        previous_hidden_dim = hidden_dim_container[0]

        # Hidden layers
        for i in range(num_hidden_layers):
            hidden_dim = hidden_dim_container[i + 1]
            layers.append(nn.Linear(previous_hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim, affine=False))
            layers.append(nn.SiLU())

            previous_hidden_dim = hidden_dim

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
