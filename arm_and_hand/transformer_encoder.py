import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_encoder_layers, dim_feedforward, dropout):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, 
            num_layers=num_encoder_layers)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, src):
        # src is of shape [seq_len, batch_size, input_dim]
        memory = self.encoder(src)
        memory = memory[-1]  # just take the last one in sequence
        # Only take the output corresponding to the last frame in the sequence
        output = self.fc(memory)  # [batch_size, input_dim] -> [batch_size, output_dim]
        return output

if __name__ == "__main__":
    input_dim = 322
    output_dim = 144
    num_heads = 7
    num_encoder_layers = 6
    dim_feedforward = 512
    dropout = 0.1

    model = TransformerEncoder(input_dim, output_dim, num_heads, num_encoder_layers, dim_feedforward, dropout)
