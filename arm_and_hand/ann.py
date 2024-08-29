import torch
import torch.nn as nn
import torch.optim as optim

class ComboLayers(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate):
        super(ComboLayers, self).__init__()

        self.linear = nn.Linear(input_dim, output_dim)
        self.act = nn.SiLU()
        self.bn = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.act(self.linear(x))
        out = self.bn(out)
        out = self.dropout(out)
        return out

# Define the neural network
class ANN(nn.Module):
    def __init__(self, input_dim=322, output_dim=144, hidden_dim=256, num_hidden_layers=2, dropout_rate=0.4):
        super(ANN, self).__init__()
        layers = []

        # Input layer
        first_layer = ComboLayers(input_dim, hidden_dim, dropout_rate)
        layers.append(first_layer)

        # Hidden layers
        for _ in range(num_hidden_layers):
            hidden_layer = ComboLayers(hidden_dim, hidden_dim, dropout_rate)
            layers.append(hidden_layer)

        pre_last_layer = ComboLayers(hidden_dim, 200, dropout_rate)
        layers.append(pre_last_layer)

        output_layer = nn.Linear(200, output_dim)
        layers.append(output_layer)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
