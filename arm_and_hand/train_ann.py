import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
from datetime import datetime
import glob
from transformer_encoder import TransformerEncoder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ann import ANN
from dataloader_ann import HandArmLandmarksDataset_ANN
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
    writer=None,
    log_seq=100):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, targets in train_dataloader:
            outputs = model(inputs)  # shape: (B, 144), B = batch_size, 144 = output_dim
            outputs = outputs.reshape(-1, 3, 48)  # shape: (B, 3, 48)
            outputs = outputs[..., :26]  # shape: (B, 3, 26), just get body and left hand to calculate loss
            targets = targets.reshape(-1, 3, 48)
            targets = targets[..., :26]
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() 

        epoch_loss = running_loss / len(train_dataloader.dataset)
        train_losses.append(epoch_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in val_dataloader:
                val_outputs = model(val_inputs)
                val_outputs = val_outputs.reshape(-1, 3, 48)  # shape: (B, 3, 48)
                val_outputs = val_outputs[..., :26]  # shape: (B, 3, 26), just get body and left hand to calculate loss
                val_targets = val_targets.reshape(-1, 3, 48)
                val_targets = val_targets[..., :26]
                val_loss += criterion(val_outputs, val_targets).item() 
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

        if epoch + 1 % log_seq == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')
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
    model_name = "ann"
    DATETIME = "{}".format(datetime.now().strftime("%Y%m%d-%H%M"))
    BASE_DIR = "/home/giakhang/dev/pose_sandbox/Hand_pose_estimation_3D/arm_and_hand/runs/{}".format(model_name)
    SAVE_DIR = os.path.join(BASE_DIR, DATETIME)
    DATA_DIR = "/home/giakhang/dev/pose_sandbox/data"  
    writer = SummaryWriter(log_dir=SAVE_DIR)

    train_paths = glob.glob(os.path.join(DATA_DIR, "*/*/fine_landmarks_{}_*.csv".format("train")))
    val_paths = glob.glob(os.path.join(DATA_DIR, "*/*/fine_landmarks_{}_*.csv".format("val")))
    body_lines = [[0,2], [0, 3], [2, 4], [3, 4]]
    lefthand_lines = [[0, 1], [1, 5], [5, 6], [5, 10], [5, 22], [10, 14], [14, 18], [18, 22], 
        [6, 7], [7, 8], [8, 9], 
        [10, 11], [11, 12], [12, 13], 
        [14, 15], [15, 16], [16, 17], 
        [18, 19], [19, 20], [20, 21], 
        [22, 23], [23, 24], [24, 25]]
    train_body_distance_thres=500
    train_leftarm_distance_thres=500
    train_lefthand_distance_thres=200
    val_body_distance_thres=500,
    val_leftarm_distance_thres=500,
    val_lefthand_distance_thres=150,

    train_dataset = HandArmLandmarksDataset_ANN(train_paths, 
        body_lines, 
        lefthand_lines, 
        train_body_distance_thres, 
        train_leftarm_distance_thres, 
        train_lefthand_distance_thres,
        filter_outlier=True,
        only_keep_frames_contain_lefthand=True)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_dataset = HandArmLandmarksDataset_ANN(val_paths,
        body_lines,
        lefthand_lines,
        val_body_distance_thres,
        val_leftarm_distance_thres,
        val_lefthand_distance_thres,
        filter_outlier=True,
        only_keep_frames_contain_lefthand=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)  

    input_dim = 322
    output_dim = 144
    hidden_dim = 256
    num_hidden_layers = 10
    dropout_rate = 0.3

    model = ANN(input_dim,
        output_dim,
        hidden_dim,
        num_hidden_layers,
        dropout_rate
    )

    pretrained_weight_path = None
    if pretrained_weight_path is not None and os.path.exists(pretrained_weight_path):
        model.load_state_dict(torch.load(pretrained_weight_path))
        print("Loaded existing model weights: ", pretrained_weight_path)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 50000
    current_time = datetime.now().strftime('%Y%m%d-%H%M')
    save_path = os.path.join(SAVE_DIR, "{}_best.path".format(model_name))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', 
        factor=math.sqrt(0.1), patience=700, verbose=True, min_lr=1e-6)
    early_stopping = EarlyStopping(patience=7000, verbose=True)

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

    writer.close()