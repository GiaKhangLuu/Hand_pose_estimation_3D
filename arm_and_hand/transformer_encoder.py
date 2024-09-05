import torch
import numpy as np
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_encoder_layers, dim_feedforward, dropout):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation="gelu")
        self.encoder = nn.TransformerEncoder(self.encoder_layer, 
            num_layers=num_encoder_layers)
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.dropout_fc1 = nn.Dropout(p=dropout)
        self.dropout_fc2 = nn.Dropout(p=dropout)
        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm1d(output_dim)

    def _create_angle_rates(self, d_model): 
        # TODO: Update document
        angles = np.arange(d_model)
        angles[1::2] = angles[0::2]
        angles = 1 / (10000 ** (angles / d_model))
        angles = np.expand_dims(angles, axis=0)
        return angles

    def _generate_positional_encoding(self, pos, d_model):
        # TODO: Update document
        angles = self._create_angle_rates(d_model)
        pos = np.expand_dims(np.arange(pos), axis=1)
        pos_angles = pos.dot(angles)
        pos_angles[:, 0::2] = np.sin(pos_angles[:, 0::2])
        pos_angles[:, 1::2] = np.cos(pos_angles[:, 1::2])
        pos_angles = np.expand_dims(pos_angles, axis=0)
  
        return torch.tensor(pos_angles, dtype=torch.float32)

    def forward(self, src):
        """
        Input:
            src: shape (seq_len, batch_size, input_dim)
        Output:
            out: shape (batch_size, output_dim)
        """
        seq_len, batch_size, input_dim = src.shape
        positional_encoding = self._generate_positional_encoding(seq_len, input_dim)  # shape: (seq_len, batch_size, input_dim)
        positional_encoding = positional_encoding.repeat(batch_size, 1, 1)  # shape: (batch_size, seq_len, input_dim)
        positional_encoding = positional_encoding.permute(1, 0, 2)  # shape: (seq_len, batch_size, input_dim)
        positional_encoding = positional_encoding.to("cuda")

        out = positional_encoding + src
        out = self.encoder(out)
        out = out[-1]  # just take the last one in sequence, shape: (batch_size, input_dim)
        if self.training:
            out = self.dropout_fc1(out)
        out = self.gelu(self.bn1(self.fc1(out)))  # (batch_size, input_dim) -> (batch_size, output_dim)
        if self.training:
            out = self.dropout_fc2(out)
        out = self.fc2(out)
        return out

if __name__ == "__main__":
    input_dim = 322
    output_dim = 144
    num_heads = 7
    num_encoder_layers = 6
    dim_feedforward = 512
    dropout = 0.1

    model = TransformerEncoder(input_dim, output_dim, num_heads, num_encoder_layers, dim_feedforward, dropout)
