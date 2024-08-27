import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network
class ANN(nn.Module):
    def __init__(self, input_dim=322, output_dim=144, hidden_dim=256, num_hidden_layers=2, dropout_rate=0.4):
        super(ANN, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_dim, 200))
        layers.append(nn.BatchNorm1d(200))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(200, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
