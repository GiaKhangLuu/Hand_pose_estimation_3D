import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
import glob
import sklearn

class HandArmLandmarksDataset_ANN(Dataset):
    def __init__(self, 
        filepaths,
        body_lines=None,
        lefthand_lines=None,
        body_distance_thres=500, 
        leftarm_distance_thres=500, 
        lefthand_distance_thres=200,
        filter_outlier=True,
        only_keep_frames_contain_lefthand=True,
        scaler=None,
        columns_to_scale=None):
        """
        We use body_lines to calculate the distance between two adjacent landmarks and filter out
        outlier with the threshold. For more info., check out notebook "data/label_for_fusing_model.ipynb".
        """

        self._inputs = []
        self._outputs = []
        self._body_lines = body_lines
        self._lefthand_lines = lefthand_lines
        self._body_distance_thres = body_distance_thres
        self._leftarm_distance_thres = leftarm_distance_thres
        self._lefthand_distance_thres = lefthand_distance_thres
        self._scaler = scaler
        self._columns_to_scale = columns_to_scale

        self._left_camera_first_intrinsic_value_idx = 144
        self._right_camera_first_lmk_value_idx = self._left_camera_first_intrinsic_value_idx + 9
        self._right_camera_first_intrinsic_value_idx = self._right_camera_first_lmk_value_idx + 144
        self._first_right_2_left_matrix_value_idx = self._right_camera_first_intrinsic_value_idx + 9

        for filepath in filepaths:
            data = pd.read_csv(filepath)
            features = data.iloc[:, 1:323].values  # Columns 2 to 323 are inputs (322 features)
            targets = data.iloc[:, 323:].values  # Columns 324 to the end are outputs (144 features)
            self._inputs.append(features)
            self._outputs.append(targets)
        self._inputs = np.asarray(
            np.concatenate(self._inputs, axis=0), dtype=np.float64)  # shape: (N, 322)
        self._outputs = np.asarray(
            np.concatenate(self._outputs, axis=0), dtype=np.float64)  # shape: (N, 144)

        if only_keep_frames_contain_lefthand:
            self._keep_frames_contain_lefthand()

        if filter_outlier: 
            assert self._body_lines is not None
            assert self._lefthand_lines is not None
            self._filter_outlier()
        
        if self._scaler:
            assert isinstance(self._scaler, sklearn.preprocessing._data.MinMaxScaler)
            assert self._columns_to_scale

            left_camera_lmks = self._inputs[:, :self._left_camera_first_intrinsic_value_idx]  # shape: (N, 144), N = #rows
            left_camera_intrinsic = self._inputs[:, self._left_camera_first_intrinsic_value_idx:self._right_camera_first_lmk_value_idx]  # shape: (N, 9), N = #rows
            right_camera_lmks = self._inputs[:, self._right_camera_first_lmk_value_idx:self._right_camera_first_intrinsic_value_idx]  # shape: (N, 144), N = #rows
            right_camera_intrinsic = self._inputs[:, self._right_camera_first_intrinsic_value_idx:self._first_right_2_left_matrix_value_idx]  # shape: (N, 9), N = #rows
            right_2_left_mat = self._inputs[:, self._first_right_2_left_matrix_value_idx:]  # shape: (N, 16), N = #rows

            left_camera_lmks = left_camera_lmks.reshape(-1, 3, 48)  # shape: (N, 3, 48)
            right_camera_lmks = right_camera_lmks.reshape(-1, 3, 48)  # shape: (N, 3, 48)
            left_camera_lmks_z = left_camera_lmks[:, -1, :]  # shape: (N, 48)
            right_camera_lmks_z = right_camera_lmks[:, -1, :]  # shape: (N, 48)

            scaled_input = np.concatenate([left_camera_lmks_z, 
                left_camera_intrinsic,
                right_camera_lmks_z,
                right_camera_intrinsic,
                right_2_left_mat], axis=1)  # shape: (N, 130)
            scaled_input = self._scaler.transform(scaled_input)
            scaled_input = pd.DataFrame(scaled_input, columns=self._columns_to_scale)
            scaled_left_camera_lmks_z = scaled_input.loc[:, "left_shoulder_cam_left_z":"right_pinky_tip_cam_left_z"]  # shape: (N, 48)
            scaled_right_camera_lmks_z = scaled_input.loc[:, "left_shoulder_cam_right_z":"right_pinky_tip_cam_right_z"]  # shape: (N, 48)
            scaled_left_camera_intr = scaled_input.loc[:, "left_camera_intrinsic_x11":"left_camera_intrinsic_x33"]  # shape: (N, 9)
            scaled_right_camera_intr = scaled_input.loc[:, "right_camera_intrinsic_x11":"right_camera_intrinsic_x33"]  # shape: (N , 9)
            scaled_right_2_left_mat = scaled_input.loc[:, "right_to_left_matrix_x11":"right_to_left_matrix_x44"]  # shape: (N, 16)

            left_camera_lmks[:, -1, :] = scaled_left_camera_lmks_z  # shape: (N, 3, 48)
            right_camera_lmks[:, -1, :] = scaled_right_camera_lmks_z  # shape: (N, 3, 48)
            left_camera_lmks = left_camera_lmks.reshape(-1, 3 * 48)  # shape: (N, 144)
            right_camera_lmks = right_camera_lmks.reshape(-1, 3 * 48)  # shape:( N, 144)

            self._inputs = np.concatenate([left_camera_lmks,
                scaled_left_camera_intr,
                right_camera_lmks,
                scaled_right_camera_intr,
                scaled_right_2_left_mat], axis=1)

    def _keep_frames_contain_lefthand(self):
        fusing_lmks = self._outputs.copy()  
        fusing_lmks = fusing_lmks.reshape(-1, 3, 48)  # shape: (N, 3, 48)
        contain_lefthand_idx = np.where(np.sum(fusing_lmks[..., 5:26], axis=(1, 2)) != 0)[0]
        self._inputs = self._inputs[contain_lefthand_idx]
        self._outputs = self._outputs[contain_lefthand_idx]
        assert self._inputs.shape[0] == self._outputs.shape[0]

    def _get_body_mask(self):
        body_distances_between_landmarks = []
        fusing_lmks = self._outputs.copy()
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

    def _get_lefthand_arm_mask(self):
        lefthand_distances_between_landmarks = []
        fusing_lmks = self._outputs.copy()
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

    def _filter_outlier(self):
        body_masks = self._get_body_mask()
        lefthand_arm_masks = self._get_lefthand_arm_mask()
        masks = np.logical_and(body_masks, lefthand_arm_masks)
        self._inputs = self._inputs[masks]
        self._outputs = self._outputs[masks]
        assert self._inputs.shape[0] == self._outputs.shape[0]

    def __len__(self):
        return self._inputs.shape[0]

    def __getitem__(self, idx):
        input_row = self._inputs[idx]
        output_row = self._outputs[idx]

        return torch.tensor(input_row, dtype=torch.float32), torch.tensor(output_row, dtype=torch.float32)

if __name__ == "__main__":
    sequence_length = 5  # Use a sequence of 5 frames
    #dataset = HandArmLandmarksDataset(inputs, outputs, sequence_length)
    #dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    DATA_DIR = "/home/giakhang/dev/pose_sandbox/data"  
    train_files = glob.glob(os.path.join(DATA_DIR, "*/*/fine_landmarks_train_*.csv"))

    HandArmLandmarksDataset(train_files, sequence_length)