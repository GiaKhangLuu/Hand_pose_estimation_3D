import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataloader import HandArmLandmarksDataset
import matplotlib.pyplot as plt
import os
from datetime import datetime
import glob
from fusing_transformer import FusingTransformer
from transformer_encoder import TransformerEncoder
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss, model, save_path):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, save_path)
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, save_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, save_path):
        torch.save(model.state_dict(), save_path)
        if self.verbose:
            print(f'Saved model with validation loss: {val_loss:.4f}')

def train_model(model, 
    train_dataloader, 
    val_dataloader, 
    criterion, 
    optimizer, 
    num_epochs=20, 
    save_path="model.pth",
    early_stopping=None,
    scheduler=None,
    writer=None):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, targets in train_dataloader:
            inputs = inputs.permute(1, 0, 2)

            optimizer.zero_grad()
            #encoder_outputs = model.encoder(inputs)
            ## Separate the encoder's output of the 5th frame and ground-truth of previous 4 frames
            #encoder_output_5th = encoder_outputs[-1].unsqueeze(0)  # [1, batch_size, output_dim]
            #gt_outputs = targets[:-1]  # [seq_len-1, batch_size, output_dim]
            ## Pass through decoder
            #predictions = model.decoder(gt_outputs, encoder_output_5th)
            #loss = criterion(predictions, targets[-1])  # Compare predictions to the ground truth of the 5th frame
            # Compute loss
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(1)

        epoch_loss = running_loss / len(train_dataloader.dataset)
        train_losses.append(epoch_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in val_dataloader:
                val_inputs = val_inputs.permute(1, 0, 2)
                #val_targets = val_targets.permute(1, 0, 2)
                #encoder_outputs = model.encoder(val_inputs)
                #encoder_output_5th = encoder_outputs[-1].unsqueeze(0)
                #gt_outputs = val_targets[:-1]
                #predictions = model.decoder(gt_outputs, encoder_output_5th)
                #loss = criterion(predictions, val_targets[-1])
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_targets).item() * val_inputs.size(1)
        val_loss = val_loss / len(val_dataloader.dataset)
        val_losses.append(val_loss)

        if scheduler is not None:
            # Adjust learning rate
            scheduler.step(val_loss)

        if writer is not None: 
            # Log losses and learning rate to TensorBoard
            writer.add_scalar('Loss/Train', epoch_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}')

        # Check early stopping criteria
        if early_stopping:
            early_stopping(val_loss, model, save_path)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break
        else:
            # Save model if validation loss decreases
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                print(f"Model saved with Validation Loss: {val_loss:.4f}")

    return train_losses, val_losses

if __name__ == "__main__":
    writer = SummaryWriter(log_dir=f'runs/fusing_transformer_{datetime.now().strftime("%Y%m%d-%H%M%S")}')

    sequence_length = 5  # Use a sequence of 5 frames
    DATA_DIR = "data"  
    train_paths = glob.glob(os.path.join(DATA_DIR, "*/*/fine_landmarks_{}_*.csv".format("train")))
    val_paths = glob.glob(os.path.join(DATA_DIR, "*/*/fine_landmarks_{}_*.csv".format("val")))
    train_dataset = HandArmLandmarksDataset(train_paths, sequence_length)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = HandArmLandmarksDataset(val_paths, sequence_length)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # No need to shuffle validation data

    input_dim = 322
    output_dim = 48 * 3  # 144
    num_encoder_heads = 7
    num_decoder_heads = 12
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 512
    dropout = 0.2
    model = TransformerEncoder(input_dim,
        output_dim,
        num_encoder_heads,
        num_encoder_layers,
        dim_feedforward,
        dropout)
    #model = FusingTransformer(input_dim, 
        #output_dim, 
        #num_encoder_heads,
        #num_decoder_heads, 
        #num_encoder_layers, 
        #num_decoder_layers, 
        #dim_feedforward, 
        #dropout)

    pretrained_weight_path = None
    if pretrained_weight_path is not None and os.path.exists(pretrained_weight_path):
        model.load_state_dict(torch.load(pretrained_weight_path))
        print("Loaded existing model weights: ", pretrained_weight_path)

    criterion = nn.L1Loss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 5000
    current_time = datetime.now().strftime('%Y-%m-%d-%H:%M')
    save_path = "transformer_encoder_best_{}.pth".format(current_time)
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', 
        factor=math.sqrt(0.1), patience=50, verbose=True, min_lr=1e-6)
    # Early stopping criteria
    early_stopping = EarlyStopping(patience=300, verbose=True)

    train_losses, val_losses = train_model(model, 
        train_dataloader, 
        val_dataloader, 
        criterion, 
        optimizer, 
        num_epochs=num_epochs, 
        save_path=save_path,
        early_stopping=early_stopping,
        scheduler=scheduler,
        writer=writer)

    # Close the TensorBoard writer
    writer.close()

    # Plot training and validation loss
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('loss_plot_{}.png'.format(current_time))  # Save the plot as a PNG file
    plt.show()
