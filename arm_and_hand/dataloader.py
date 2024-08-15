import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class HandArmLandmarksDataset(Dataset):
    def __init__(self, filepaths, sequence_length):
        #self.inputs = inputs
        #self.outputs = outputs
        self.sequence_length = sequence_length
        self.seq_len_each_file = []

        self.inputs = []
        self.outputs = []

        for filepath in filepaths:
            print("---------------")
            print(filepath)
            data = pd.read_csv(filepath)
            
            # Extract inputs and outputs
            features = data.iloc[:, 1:323].values  # Columns 2 to 323 are inputs (322 features)
            targets = data.iloc[:, 323:].values  # Columns 324 to 476 are outputs (153 features)

            # Add to the lists
            self.inputs.append(features)
            self.outputs.append(targets)

            #seq_len = features.shape[0] - self.sequence_length  
            #self.seq_len_each_file.append(seq_len)
        
        self.inputs = np.concatenate(self.inputs, axis=0)
        self.outputs = np.concatenate(self.outputs, axis=0)

    def __len__(self):
        #return sum(self.seq_len_each_file)
        return len(self.inputs) - self.sequence_length + 1

    def __getitem__(self, idx):
        #print(idx)
        #last_index = 0
        #for i, (input_df, output_df) in enumerate(zip(self.inputs, self.outputs)):
            #if idx > input_df.shape[0] + last_index:
                #last_index += self.seq_len_each_file[i]
            #else:
                #selected_idx = idx - last_index
                #input_seq = input_df[selected_idx:selected_idx+self.sequence_length]
                #output_target = output_df[selected_idx + self.sequence_length - 1]  # Only the 5th frame's output
                #return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(output_target, dtype=torch.float32)
        #return None, None
        input_seq = self.inputs[idx:idx + self.sequence_length]
        output_seq = self.outputs[idx + self.sequence_length - 1]
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(output_seq, dtype=torch.float32)

if __name__ == "__main__":
    sequence_length = 5  # Use a sequence of 5 frames
    #dataset = HandArmLandmarksDataset(inputs, outputs, sequence_length)
    #dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    DATA_DIR = "/home/giakhang/dev/pose_sandbox/Hand_pose_estimation_3D/arm_and_hand/data"  
    train_paths = []

    # Walk through the directory and its subdirectories
    for d, _, files in os.walk(DATA_DIR):
        for file in files:
            if "train" in file and file.endswith('.csv'):
                file_path = os.path.join(d, file)
                #df = pd.read_csv(file_path)
                train_paths.append(file_path)

    HandArmLandmarksDataset(train_paths, sequence_length)