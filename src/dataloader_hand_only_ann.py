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
from dataloader_ann import HandArmLandmarksDataset_ANN

class HandLandmarksDataset_ANN(HandArmLandmarksDataset_ANN):
    def __init__(
        self, 
        filepaths,
        fusing_landmark_dictionary,
        body_lines=None,
        lefthand_lines=None,
        body_distance_thres=500, 
        leftarm_distance_thres=500, 
        lefthand_distance_thres=200,
        filter_outlier=True,
        only_keep_frames_contain_lefthand=True,
        cvt_normalized_xy_to_XY=True,
        use_fused_thumb_as_input=False,
        use_thumb_as_anchor=False,
        input_scaler=None, 
        output_scaler=None
    ):
        """
        We use body_lines to calculate the distance between two adjacent landmarks and filter out
        outlier with the threshold. For more info., check out notebook "data/label_for_fusing_model.ipynb".
        """

        super().__init__(
            filepaths=filepaths,
            fusing_landmark_dictionary=fusing_landmark_dictionary,
            body_lines=body_lines,
            lefthand_lines=lefthand_lines,
            body_distance_thres=body_distance_thres,
            leftarm_distance_thres=leftarm_distance_thres,
            lefthand_distance_thres=lefthand_distance_thres,
            filter_outlier=filter_outlier,
            only_keep_frames_contain_lefthand=only_keep_frames_contain_lefthand,
            scaler=False,
            cvt_normalized_xy_to_XY=cvt_normalized_xy_to_XY,
            use_fused_thumb_as_input=False
        )

        self._input_scaler = input_scaler
        self._output_scaler = output_scaler

        self._inputs = self._inputs.reshape(-1, 3, 48 * 2)   # (N, 3, 48 * 2 + len(self._thumb_idx))

        self._full_outputs = self._outputs
        self._outputs = self._outputs.reshape(-1, 3, 48)  # (N, 3, 48)

        num_landmarks = len(self._fusing_landmark_dictionary)
        inputs_from_left_cam = self._inputs[..., :num_landmarks]
        inputs_from_right_cam = self._inputs[..., num_landmarks:num_landmarks * 2]
        #hybrid_lmks = self._inputs[..., num_landmarks * 2:]

        assert inputs_from_left_cam.shape[-1] == num_landmarks
        assert inputs_from_right_cam.shape[-1] == num_landmarks
        #assert hybrid_lmks.shape[-1] == len(self._thumb_idx)
        
        left_wrist_idx = 5
        num_hand_lmks = 21
        body_inputs_from_left_cam = inputs_from_left_cam[..., :left_wrist_idx]
        body_inputs_from_right_cam = inputs_from_right_cam[..., :left_wrist_idx]
        left_hand_inputs_from_left_cam = inputs_from_left_cam[..., left_wrist_idx:left_wrist_idx + num_hand_lmks]
        left_hand_inputs_from_right_cam = inputs_from_right_cam[..., left_wrist_idx:left_wrist_idx + num_hand_lmks]
        #left_hand_hybrid_lmks = hybrid_lmks[..., left_wrist_idx:]

        self._inputs = np.concatenate([
            left_hand_inputs_from_left_cam,
            left_hand_inputs_from_right_cam], axis=-1)  # (N, 3, 26 + 26)
        self._body_inputs = np.concatenate([
            body_inputs_from_left_cam,
            body_inputs_from_right_cam
        ], axis=-1)

        self._body_outputs = self._outputs[..., :left_wrist_idx]  # (N, 3, 5)
        self._outputs = self._outputs[..., left_wrist_idx:left_wrist_idx + num_hand_lmks]  # (N, 3, 26)

        if use_thumb_as_anchor:
            self._inputs = np.concatenate([self._inputs, self._body_outputs], axis=-1)  # (N, 3, 26 + 26 + 5)

        self._inputs = self._inputs.reshape(self._inputs.shape[0], -1)
        self._body_inputs = self._body_inputs.reshape(self._body_inputs.shape[0], -1)
        
        self._outputs = self._outputs.reshape(-1, 3 * num_hand_lmks)
        self._body_outputs = self._body_outputs.reshape([self._body_outputs.shape[0], -1])

        if self._input_scaler:
            assert isinstance(self._input_scaler, LandmarksScaler)
            self._inputs = self._input_scaler(self._inputs)

        if self._output_scaler:
            assert isinstance(self._output_scaler, LandmarksScaler)
            self._outputs = self._output_scaler(self._outputs)