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
from dataloader_hand_only_ann import HandLandmarksDataset_ANN

class HandLandmarksDataset_ANN_With_Anchors(HandLandmarksDataset_ANN):
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
            cvt_normalized_xy_to_XY=cvt_normalized_xy_to_XY,
            use_fused_thumb_as_input=use_fused_thumb_as_input,
            use_thumb_as_anchor=False,
            input_scaler=None,
            output_scaler=None
        )

        self._inputs = self._inputs.reshape(self._inputs.shape[0], 3, -1)  # (N, 3, (21 * 3 * 2) + (5 * 3))
        self._outputs = self._outputs.reshape(-1, 3, 21)
        self._body_outputs = self._body_outputs.reshape(-1, 3, 5)

        self.hand_anchor_indexes = [0, 5, 9]
        self._outputs_wrist = self._outputs[..., self.hand_anchor_indexes[0]]  # (N, 3)   
        self._outputs_index_finger_mcp = self._outputs[..., self.hand_anchor_indexes[1]]  # (N, 3)
        self._outputs_middle_finger_mcp = self._outputs[..., self.hand_anchor_indexes[2]]  # (N, 3)

        self._outputs = np.delete(self._outputs, self.hand_anchor_indexes, -1)  # (N, 3, 18)

        self._hand_anchors = np.concatenate([  # (N, 3, 3)
            self._outputs_wrist[..., None],
            self._outputs_index_finger_mcp[..., None],
            self._outputs_middle_finger_mcp[..., None],
        ], axis=-1)  

        self._inputs = np.concatenate([self._inputs, self._hand_anchors], axis=-1)  # (N, 3, (21 * 3 * 2) + (5 * 3) + (3 * 3))

        self._inputs = self._inputs.reshape(self._inputs.shape[0], -1)
        self._outputs = self._outputs.reshape(self._outputs.shape[0], -1)

        self._input_scaler = input_scaler
        self._output_scaler = output_scaler

        if self._input_scaler:
            assert isinstance(self._input_scaler, LandmarksScaler)
            self._inputs = self._input_scaler(self._inputs)

        if self._output_scaler:
            assert isinstance(self._output_scaler, LandmarksScaler)
            self._outputs = self._output_scaler(self._outputs)
