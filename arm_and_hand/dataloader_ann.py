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
from landmarks_scaler import LandmarksScaler
from csv_writer import fusion_csv_columns_name
from utilities import xyZ_to_XYZ

class HandArmLandmarksDataset_ANN(Dataset):
    def __init__(self, 
        filepaths,
        fusing_landmark_dictionary,
        body_lines=None,
        lefthand_lines=None,
        body_distance_thres=500, 
        leftarm_distance_thres=500, 
        lefthand_distance_thres=200,
        filter_outlier=True,
        only_keep_frames_contain_lefthand=True,
        scaler=None,
        cvt_normalized_xy_to_XY=True,
        use_fused_thumb_as_input=False):
        """
        We use body_lines to calculate the distance between two adjacent landmarks and filter out
        outlier with the threshold. For more info., check out notebook "data/label_for_fusing_model.ipynb".
        """

        self._inputs = []
        self._outputs = []
        self._left_cam_intrinsic_container = []
        self._right_cam_intrinsic_container = []
        self._right_to_left_matrix_container = []
        self._frame_size_containter = []
        self._calibrated_frame_size_container= []
        self._body_lines = body_lines
        self._lefthand_lines = lefthand_lines
        self._body_distance_thres = body_distance_thres
        self._leftarm_distance_thres = leftarm_distance_thres
        self._lefthand_distance_thres = lefthand_distance_thres
        self._scaler = scaler
        self._cvt_normalized_xy_to_XY = cvt_normalized_xy_to_XY
        self._use_fused_thumb_as_input = use_fused_thumb_as_input
        self._fusing_landmark_dictionary = fusing_landmark_dictionary

        self._thumb_landmarks = ["left shoulder", "left hip", "right shoulder", "right hip", 
            "left elbow", "WRIST", 
            "THUMB_CMC", "INDEX_FINGER_MCP", "MIDDLE_FINGER_MCP", "RING_FINGER_MCP", "PINKY_MCP",
            "THUMB_TIP", "INDEX_FINGER_TIP", "MIDDLE_FINGER_TIP", "RING_FINGER_TIP", "PINKY_TIP"]
        self._thumb_idx = [self._fusing_landmark_dictionary.index(lmks) for lmks in self._thumb_landmarks]

        for filepath in filepaths:
            data = pd.read_csv(filepath)
            features = data.loc[:, "left_shoulder_cam_left_x":"right_to_left_matrix_x44"].values  # Columns 2 to 323 are inputs (322 features)
            targets = data.loc[:, "left_shoulder_output_x":"right_pinky_tip_output_z"].values  # Columns 324 to the end are outputs (144 features)
            
            left_cam_intrinsics = np.array(data.loc[:, 
                                                    "left_camera_intrinsic_x11":"left_camera_intrinsic_x33"].values).reshape(-1, 3, 3)
            right_cam_intrinsics = np.array(data.loc[:,
                                                     "right_camera_intrinsic_x11":"right_camera_intrinsic_x33"].values).reshape(-1, 3, 3)
            right_to_left_mats = np.array(data.loc[:,
                                                   "right_to_left_matrix_x11":"right_to_left_matrix_x44"].values).reshape(-1, 4, 4)
            frame_size = data.loc[:, "frame_width":"frame_height"].values
            calibrated_frame_size = data.loc[:, "frame_calibrated_width":"frame_calibrated_height"].values 
            
            self._inputs.append(features)
            self._outputs.append(targets)
            self._left_cam_intrinsic_container.append(left_cam_intrinsics)
            self._right_cam_intrinsic_container.append(right_cam_intrinsics)
            self._right_to_left_matrix_container.append(right_to_left_mats)
            self._frame_size_containter.append(frame_size)
            self._calibrated_frame_size_container.append(calibrated_frame_size)
        self._inputs = np.asarray(
            np.concatenate(self._inputs, axis=0), dtype=np.float64)  # shape: (N, 322)
        self._outputs = np.asarray(
            np.concatenate(self._outputs, axis=0), dtype=np.float64)  # shape: (N, 144)
        self._left_cam_intrinsic_container = np.asarray(
            np.concatenate(self._left_cam_intrinsic_container, axis=0), dtype=np.float64)  # shape: (N, 3, 3)
        self._right_cam_intrinsic_container = np.asarray(
            np.concatenate(self._right_cam_intrinsic_container, axis=0), dtype=np.float64)  # shape: (N, 3, 3)
        self._right_to_left_matrix_container = np.asarray(
            np.concatenate(self._right_to_left_matrix_container, axis=0), dtype=np.float64)  # shape: (N, 4, 4) 
        self._frame_size_containter = np.asarray(
            np.concatenate(self._frame_size_containter, axis=0), dtype=np.float64)  # shape: (N, 2)
        self._calibrated_frame_size_container = np.asarray(
            np.concatenate(self._calibrated_frame_size_container, axis=0), dtype=np.float64)  # shape: (N, 2)    
        
        assert self._inputs.shape[1] == 322
        assert self._outputs.shape[1] == 144

        if only_keep_frames_contain_lefthand:
            self._keep_frames_contain_lefthand()

        if filter_outlier: 
            assert self._body_lines is not None
            assert self._lefthand_lines is not None
            self._filter_outlier()
            
        # Scale intrinsic
        # -> Dont need to scale intrinsic matrix anymore, because these matrices were scaled before.
        #self._left_cam_intrinsic_container = scale_intrinsic_by_res(self._left_cam_intrinsic_container,
                                                                    #self._calibrated_frame_size_container[..., ::-1],
                                                                    #self._frame_size_containter[..., ::-1])
        #self._right_cam_intrinsic_container = scale_intrinsic_by_res(self._right_cam_intrinsic_container,
                                                                        #self._calibrated_frame_size_container[..., ::-1],
                                                                        #self._frame_size_containter[..., ::-1])
            
        if self._cvt_normalized_xy_to_XY:
            input_df = pd.DataFrame(self._inputs, columns=fusion_csv_columns_name[1:fusion_csv_columns_name.index("left_shoulder_output_x")])
            left_camera_first_lmks = "left_shoulder_cam_left_x"
            left_camera_last_lmks = "right_pinky_tip_cam_left_z"
            left_camera_first_idx = fusion_csv_columns_name.index(left_camera_first_lmks)
            left_camera_last_idx = fusion_csv_columns_name.index(left_camera_last_lmks)
            left_camera_lmks_columns_name = fusion_csv_columns_name[left_camera_first_idx:left_camera_last_idx+1]
            left_camera_lmks = input_df.loc[:, left_camera_lmks_columns_name].values
            
            right_camera_first_lmks = "left_shoulder_cam_right_x"
            right_camera_last_lmks = "right_pinky_tip_cam_right_z"
            right_camera_first_idx = fusion_csv_columns_name.index(right_camera_first_lmks)
            right_camera_last_idx = fusion_csv_columns_name.index(right_camera_last_lmks)
            right_camera_lmks_columns_name = fusion_csv_columns_name[right_camera_first_idx:right_camera_last_idx+1]
            right_camera_lmks = input_df.loc[:, right_camera_lmks_columns_name].values
            
            # Unnormalized 
            left_camera_unnormalized_lmks = left_camera_lmks.copy()
            left_camera_unnormalized_lmks = left_camera_unnormalized_lmks.reshape(-1, 3, 48)  # shape: (N, 3, 48) 
            left_camera_unnormalized_lmks[:, 0, :] *= self._frame_size_containter[:, 0][:, None]
            left_camera_unnormalized_lmks[:, 1, :] *= self._frame_size_containter[:, 1][:, None]
            
            right_camera_unnormalized_lmks = right_camera_lmks.copy()
            right_camera_unnormalized_lmks = right_camera_unnormalized_lmks.reshape(-1, 3, 48)  # shape: (N, 3, 48) 
            right_camera_unnormalized_lmks[:, 0, :] *=  self._frame_size_containter[:, 0][:, None]
            right_camera_unnormalized_lmks[:, 1, :] *=  self._frame_size_containter[:, 1][:, None]
            
            # Convert xyZ to XYZ 
            left_camera_xyZ = np.transpose(left_camera_unnormalized_lmks, (0, 2, 1))  # shape: (N, 48, 3)
            right_camera_xyZ = np.transpose(right_camera_unnormalized_lmks, (0, 2, 1))  # shape: (N, 48, 3)
            left_camera_XYZ = xyZ_to_XYZ(left_camera_xyZ, self._left_cam_intrinsic_container)
            right_camera_XYZ = xyZ_to_XYZ(right_camera_xyZ, self._right_cam_intrinsic_container) 
            
            left_camera_XYZ = np.transpose(left_camera_XYZ, (0, 2, 1))  # shape: (N, 3, 48)
            right_camera_XYZ = np.transpose(right_camera_XYZ, (0, 2, 1))  # shape: (N, 3, 48)
            left_camera_XYZ = left_camera_XYZ.reshape(-1, 3 * 48)
            right_camera_XYZ = right_camera_XYZ.reshape(-1, 3 * 48)
            
            self._inputs = np.concatenate([left_camera_XYZ, right_camera_XYZ], axis=1)

        if self._use_fused_thumb_as_input:
            fusing_lmks = self._outputs.copy()  
            fusing_lmks = fusing_lmks.reshape(-1, 3, 48)  # shape: (N, 3, 48)
            thumb_XYZ_landmarks = fusing_lmks[..., self._thumb_idx]  # shape: (N, 3, 4)
            self._inputs = self._inputs.reshape(-1, 3, 48 * 2)  # shape: (N, 3, 48 * 2)
            self._inputs = np.concatenate([self._inputs, thumb_XYZ_landmarks], axis=-1)  # shape: (N, 3, 48 * 2 + 4)
            self._inputs = self._inputs.reshape(-1, 3 * (48 * 2 + len(self._thumb_landmarks)))
            #self._outputs = self._outputs.reshape(-1, 3, 48)
            #self._outputs[..., thumb_idx] *= 0.0
            #self._outputs = self._outputs.reshape(-1, 3 * 48)
        
        if self._scaler:
            assert isinstance(self._scaler, LandmarksScaler)
            self._inputs = self._scaler(self._inputs)


    def _keep_frames_contain_lefthand(self):
        fusing_lmks = self._outputs.copy()  
        fusing_lmks = fusing_lmks.reshape(-1, 3, 48)  # shape: (N, 3, 48)
        contain_lefthand_idx = np.where(np.sum(fusing_lmks[..., 5:26], axis=(1, 2)) != 0)[0]
        self._inputs = self._inputs[contain_lefthand_idx]
        self._outputs = self._outputs[contain_lefthand_idx]
        self._left_cam_intrinsic_container = self._left_cam_intrinsic_container[contain_lefthand_idx]
        self._right_cam_intrinsic_container = self._right_cam_intrinsic_container[contain_lefthand_idx]
        self._right_to_left_matrix_container = self._right_to_left_matrix_container[contain_lefthand_idx]
        self._frame_size_containter = self._frame_size_containter[contain_lefthand_idx]
        self._calibrated_frame_size_container = self._calibrated_frame_size_container[contain_lefthand_idx]
        assert (self._inputs.shape[0] == 
                self._outputs.shape[0] == 
                self._left_cam_intrinsic_container.shape[0] == 
                self._right_cam_intrinsic_container.shape[0] == 
                self._right_to_left_matrix_container.shape[0] ==
                self._frame_size_containter.shape[0] == 
                self._calibrated_frame_size_container.shape[0])

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
        self._left_cam_intrinsic_container = self._left_cam_intrinsic_container[masks]
        self._right_cam_intrinsic_container = self._right_cam_intrinsic_container[masks]
        self._right_to_left_matrix_container = self._right_to_left_matrix_container[masks]
        self._frame_size_containter = self._frame_size_containter[masks]
        self._calibrated_frame_size_container = self._calibrated_frame_size_container[masks]
        assert (self._inputs.shape[0] == 
                self._outputs.shape[0] ==
                self._left_cam_intrinsic_container.shape[0] == 
                self._right_cam_intrinsic_container.shape[0] == 
                self._right_to_left_matrix_container.shape[0] == 
                self._frame_size_containter.shape[0] == 
                self._calibrated_frame_size_container.shape[0])

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