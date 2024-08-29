import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import glob
import numpy as np
from scipy.spatial.distance import euclidean

class HandArmLandmarksDataset_Transformer_Encoder(Dataset):
    def __init__(self, 
        filepaths, 
        sequence_length,
        body_lines=None,
        lefthand_lines=None,
        body_distance_thres=500, 
        leftarm_distance_thres=500, 
        lefthand_distance_thres=200,
        filter_outlier=True,
        only_keep_frames_contain_lefthand=True):
        """
        We use body_lines to calculate the distance between two adjacent landmarks and filter out
        outlier with the threshold. For more info., check out notebook "data/label_for_fusing_model.ipynb".
        """

        self._body_lines = body_lines
        self._lefthand_lines = lefthand_lines
        self._body_distance_thres = body_distance_thres
        self._leftarm_distance_thres = leftarm_distance_thres
        self._lefthand_distance_thres = lefthand_distance_thres

        self.sequence_length = sequence_length
        self.checkpoint_each_file = [0]

        self.inputs = []
        self.outputs = []

        for filepath in filepaths:
            data = pd.read_csv(filepath)
            # Extract inputs and outputs
            features = data.iloc[:, 1:323].values  # Columns 2 to 323 are inputs (322 features)
            targets = data.iloc[:, 323:].values  # Columns 324 to the end are outputs (144 features)
            
            if only_keep_frames_contain_lefthand:
                features, targets = self._keep_frames_contain_lefthand(features, targets)

            if filter_outlier: 
                assert self._body_lines is not None
                assert self._lefthand_lines is not None
                features, targets = self._filter_outlier(features, targets)

            num_rows = features.shape[0]
            if num_rows < sequence_length:
                continue

            self.inputs.append(features)
            self.outputs.append(targets)
            num_seq = num_rows - self.sequence_length + 1
            checkpoint = self.checkpoint_each_file[-1] + num_seq
            self.checkpoint_each_file.append(checkpoint)
        self.checkpoint_each_file = self.checkpoint_each_file[1:]

    def _keep_frames_contain_lefthand(self, inputs, targets):
        fusing_lmks = targets.copy()  
        fusing_lmks = fusing_lmks.reshape(-1, 3, 48)  # shape: (N, 3, 48)
        contain_lefthand_idx = np.where(np.sum(fusing_lmks[..., 5:26], axis=(1, 2)) != 0)[0]
        inputs = inputs[contain_lefthand_idx]
        targets = targets[contain_lefthand_idx]
        assert inputs.shape[0] == targets.shape[0]

        return inputs, targets

    def _get_body_mask(self, targets):
        body_distances_between_landmarks = []
        fusing_lmks = targets.copy()
        fusing_lmks = fusing_lmks.reshape(-1, 3, 48)  # shape: (N, 3, 48)
        fusing_lmks = np.transpose(fusing_lmks, (0, 2, 1))  # shape: (N, 48, 3)
        for prev_idx, next_idx in self._body_lines:
            distances_between_landmarks = []
            for i in range(fusing_lmks.shape[0]):
                x = fusing_lmks[i, prev_idx, :] 
                y = fusing_lmks[i, next_idx, :]
                dis = euclidean(x, y)
                distances_between_landmarks.append(dis)
            body_distances_between_landmarks.append(distances_between_landmarks)
        body_distances_between_landmarks = np.array(body_distances_between_landmarks)  # shape: (4, N), N = #frames, 4 = #lines
        body_distances_between_landmarks = body_distances_between_landmarks.T  # shape: (N, 4)
        body_masks = body_distances_between_landmarks < self._body_distance_thres  # shape: (N, 4)
        body_masks = np.all(body_masks, axis=1)  # shape: (N)
        return body_masks

    def _get_lefthand_arm_mask(self, targets):
        lefthand_distances_between_landmarks = []
        fusing_lmks = targets.copy()
        fusing_lmks = fusing_lmks.reshape(-1, 3, 48)  # shape: (N, 3, 48)
        fusing_lmks = np.transpose(fusing_lmks, (0, 2, 1))  # shape: (N, 48, 3)
        for prev_idx, next_idx in self._lefthand_lines:
            dis_between_landmarks = []
            for i in range(fusing_lmks.shape[0]):
                x = fusing_lmks[i, prev_idx, :] 
                y = fusing_lmks[i, next_idx, :]
                dis = euclidean(x, y)
                dis_between_landmarks.append(dis)
            lefthand_distances_between_landmarks.append(dis_between_landmarks)
        lefthand_distances_between_landmarks = np.array(lefthand_distances_between_landmarks)  # shape: (23, N), N = #frames, 23 = #lines
        lefthand_distances_between_landmarks = lefthand_distances_between_landmarks.T  # shape: (N, 23)
        leftarm_masks = lefthand_distances_between_landmarks[:, :2] < self._leftarm_distance_thres  # shape: (N, 2)
        lefthand_masks = lefthand_distances_between_landmarks[:, 2:] < self._lefthand_distance_thres  # shape:(N, 21)
        lefthand_arm_masks = np.concatenate([leftarm_masks, lefthand_masks], axis=1)  # shape: (N, 23)
        lefthand_arm_masks = np.all(lefthand_arm_masks, axis=1)  # shape: (N)
        return lefthand_arm_masks

    def _filter_outlier(self, inputs, targets):
        body_masks = self._get_body_mask(targets)
        lefthand_arm_masks = self._get_lefthand_arm_mask(targets)
        masks = np.logical_and(body_masks, lefthand_arm_masks)
        inputs = inputs[masks]
        targets = targets[masks]
        assert inputs.shape[0] == targets.shape[0]

        return inputs, targets

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