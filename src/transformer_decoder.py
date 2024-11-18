import torch
import torch.nn as nn
from transformer_encoder import TransformerEncoder

class TransformerDecoder(nn.Module):
    def __init__(self, output_dim, num_heads, num_layers, dim_feedforward, dropout):
        super(TransformerDecoder, self).__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=output_dim, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(output_dim, output_dim)
    
    def forward(self, tgt, memory):
        # tgt is the previous ground-truth outputs [seq_len, batch_size, output_dim]
        # memory is the encoded output of the 5th frame [seq_len=1, batch_size, output_dim]

        # Pass through the Transformer decoder
        output = self.decoder(tgt, memory)
        output = self.fc(output[-1])  # Only take the last output from the decoder
        return output

if __name__ == "__main__":
    # Initialize models
    input_dim = 322
    output_dim = 48 * 3  # 144
    num_heads = 12
    num_encoder_layers = 3
    num_decoder_layers = 3
    dim_feedforward = 512
    dropout = 0.2

    encoder = TransformerEncoder(input_dim, output_dim, 7, num_encoder_layers, dim_feedforward, dropout)
    decoder = TransformerDecoder(output_dim, 12, num_decoder_layers, dim_feedforward, dropout)

    # Example inputs
    src = torch.randn(5, 1, input_dim)  # [seq_len=5, batch_size, input_dim]
    gt_outputs = torch.randn(4, 1, output_dim)  # [seq_len=4, batch_size, output_dim]

    # Get encoded outputs from encoder
    encoder_outputs = encoder(src)

    # Pass the 5th frame through the decoder with ground-truth outputs
    encoder_output_5th = encoder_outputs[-1].unsqueeze(0)  # [seq_len=1, batch_size, output_dim]
    final_output = decoder(gt_outputs, encoder_output_5th)

    print(final_output.shape)  # Should be [batch_size, output_dim]