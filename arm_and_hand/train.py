import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformer_encoder import TransformerEncoder
from dataloader import HandArmLandmarksDataset
import matplotlib.pyplot as plt
import os
from datetime import datetime

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
    early_stopping=None):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, targets in train_dataloader:
            inputs = inputs.permute(1, 0, 2)  # Transpose to [seq_len, batch_size, input_dim]
            targets = targets  # [batch_size, output_dim]

            optimizer.zero_grad()
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
                val_inputs = val_inputs.permute(1, 0, 2)  # Transpose to [seq_len, batch_size, input_dim]
                val_targets = val_targets  # [batch_size, output_dim]

                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_targets).item() * val_inputs.size(1)

        val_loss = val_loss / len(val_dataloader.dataset)
        val_losses.append(val_loss)
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
    sequence_length = 5  # Use a sequence of 5 frames

    DATA_DIR = "/home/giakhang/dev/pose_sandbox/Hand_pose_estimation_3D/arm_and_hand/data"  
    train_paths = []
    val_paths = []

    # Walk through the directory and its subdirectories
    for d, _, files in os.walk(DATA_DIR):
        for file in files:
            if "train" in file and file.endswith('.csv'):
                file_path = os.path.join(d, file)
                train_paths.append(file_path)
            if "val" in file and file.endswith(".csv"):
                file_path = os.path.join(d, file)
                val_paths.append(file_path)

    train_dataset = HandArmLandmarksDataset(train_paths, sequence_length)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = HandArmLandmarksDataset(val_paths, sequence_length)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # No need to shuffle validation data

    input_dim = 322
    output_dim = 153
    num_heads = 7
    num_encoder_layers = 6
    dim_feedforward = 512
    dropout = 0.3

    model = TransformerEncoder(input_dim, output_dim, num_heads, num_encoder_layers, dim_feedforward, dropout)

    pretrained_weight_path = "/home/giakhang/dev/pose_sandbox/Hand_pose_estimation_3D/arm_and_hand/best_model_2024-08-15-10:37.pth"
    if os.path.exists(pretrained_weight_path):
        model.load_state_dict(torch.load(pretrained_weight_path))
        print("Loaded existing model weights: ", pretrained_weight_path)

    criterion = nn.L1Loss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 1000
    current_time = datetime.now().strftime('%Y-%m-%d-%H:%M')
    save_path = "best_model_{}.pth".format(current_time)

    # Early stopping criteria
    early_stopping = EarlyStopping(patience=300, verbose=True)

    train_losses, val_losses = train_model(model, 
        train_dataloader, 
        val_dataloader, 
        criterion, 
        optimizer, 
        num_epochs=num_epochs, 
        save_path=save_path,
        early_stopping=early_stopping)

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
