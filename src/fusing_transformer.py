import torch 
import torch.nn as nn
from transformer_encoder import TransformerEncoder
from transformer_decoder import TransformerDecoder

class FusingTransformer(nn.Module):
    def __init__(self, 
        input_dim, 
        output_dim, 
        num_encoder_heads, 
        num_decoder_heads,
        num_encoder_layers, 
        num_decoder_layers, 
        dim_feedforward, 
        dropout):
        super(FusingTransformer, self).__init__()
        self.encoder = TransformerEncoder(input_dim, 
            output_dim, 
            num_encoder_heads, 
            num_encoder_layers, 
            dim_feedforward, 
            dropout)
        self.decoder = TransformerDecoder(output_dim, 
            num_decoder_heads, 
            num_decoder_layers, 
            dim_feedforward, 
            dropout)

    def forward(self, src, tgt):
        # Encode the 5th frame from the input sequence
        encoder_outputs = self.encoder(src)
        encoder_output_5th = encoder_outputs[-1].unsqueeze(0)  # [1, batch_size, output_dim]

        # Decode using the previous 4 ground-truth outputs and the encoded 5th frame
        output = self.decoder(tgt, encoder_output_5th)
        return output

if __name__ == "__main__":
    # Example usage
    input_dim = 322
    output_dim = 48 * 3  # 144
    num_encoder_heads = 7
    num_decoder_heads = 12
    num_encoder_layers = 3
    num_decoder_layers = 3
    dim_feedforward = 512
    dropout = 0.2

    # Initialize the full model
    model = FusingTransformer(input_dim, 
        output_dim, 
        num_encoder_heads,
        num_decoder_heads, 
        num_encoder_layers, 
        num_decoder_layers, 
        dim_feedforward, 
        dropout)

    # Example input and target tensors
    src = torch.randn(5, 1, input_dim)  # 5 frames, batch size 1, input dim 322
    gt_outputs = torch.randn(4, 1, output_dim)  # 4 frames of ground-truth outputs, batch size 1, output dim 144

    # Forward pass
    output = model(src, gt_outputs)
    print(output.shape)  # Should be [batch_size, output_dim]