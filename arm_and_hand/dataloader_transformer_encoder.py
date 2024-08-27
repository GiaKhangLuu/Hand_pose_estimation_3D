import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import glob

class HandArmLandmarksDataset_Encoder(Dataset):
    def __init__(self, filepaths, sequence_length):
        self.sequence_length = sequence_length
        self.checkpoint_each_file = [0]

        self.inputs = []
        self.outputs = []

        for filepath in filepaths:
            data = pd.read_csv(filepath)
            # Extract inputs and outputs
            features = data.iloc[:, 1:323].values  # Columns 2 to 323 are inputs (322 features)
            targets = data.iloc[:, 323:].values  # Columns 324 to the end are outputs (144 features)
            num_rows = features.shape[0]
            if num_rows < sequence_length:
                continue
            self.inputs.append(features)
            self.outputs.append(targets)
            num_seq = num_rows - self.sequence_length + 1
            checkpoint = self.checkpoint_each_file[-1] + num_seq
            self.checkpoint_each_file.append(checkpoint)
        self.checkpoint_each_file = self.checkpoint_each_file[1:]

    def __len__(self):
        return self.checkpoint_each_file[-1]

    def __getitem__(self, idx):
        selected_df_idx = 0
        assert idx < self.checkpoint_each_file[-1]
        # Linear seach to find file based on idx
        for i, ckpt in enumerate(self.checkpoint_each_file):
            if idx < ckpt:
                selected_df_idx = i
                break
        offset_idx = self.checkpoint_each_file[selected_df_idx - 1] if selected_df_idx > 0 else 0
        idx_in_df = idx - offset_idx
        inputs_df, output_df = self.inputs[selected_df_idx], self.outputs[selected_df_idx]
        input_seq = inputs_df[idx_in_df:idx_in_df + self.sequence_length]
        output_seq = output_df[idx_in_df + self.sequence_length - 1]
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(output_seq, dtype=torch.float32)

if __name__ == "__main__":
    sequence_length = 5  # Use a sequence of 5 frames
    #dataset = HandArmLandmarksDataset(inputs, outputs, sequence_length)
    #dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    DATA_DIR = "/home/giakhang/dev/pose_sandbox/data"  
    train_files = glob.glob(os.path.join(DATA_DIR, "*/*/fine_landmarks_train_*.csv"))

    HandArmLandmarksDataset(train_files, sequence_length)