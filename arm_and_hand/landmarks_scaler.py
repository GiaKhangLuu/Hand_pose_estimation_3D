import sklearn
import joblib
import numpy as np
import pandas as pd

class LandmarksScaler():
    """
    Currently, use MinMaxScaler from sklearn to scale z_values, intrinsic matrices
    """

    def __init__(self, columns_to_scale, scaler_path=None):
        assert scaler_path is not None 

        self._columns_to_scale = columns_to_scale
        self._minmax_scaler = joblib.load(scaler_path)
        self._num_features = 322

        assert isinstance(self._minmax_scaler, sklearn.preprocessing._data.MinMaxScaler)

        # These index values based on the values writed to .csv file. Check a module which saves landmarks for 
        # more information.
        self._left_camera_first_intrinsic_value_idx = 144  
        self._right_camera_first_lmk_value_idx = self._left_camera_first_intrinsic_value_idx + 9
        self._right_camera_first_intrinsic_value_idx = self._right_camera_first_lmk_value_idx + 144
        self._first_right_2_left_matrix_value_idx = self._right_camera_first_intrinsic_value_idx + 9

    def __call__(self, landmarks_input):
        """
        Input:
            landmarks_input (np.array): landmarks input to scale, shape = (N, self._num_features), N = #data
        Output:
            scaled_landmarks_input (np.array): shape = (N, self._num_features)
        """
        assert landmarks_input.ndim == 2
        assert landmarks_input.shape[1] == self._num_features

        left_camera_lmks = landmarks_input[:, :self._left_camera_first_intrinsic_value_idx]  # shape: (N, 144), N = #rows
        left_camera_intrinsic = landmarks_input[:, self._left_camera_first_intrinsic_value_idx:self._right_camera_first_lmk_value_idx]  # shape: (N, 9), N = #rows
        right_camera_lmks = landmarks_input[:, self._right_camera_first_lmk_value_idx:self._right_camera_first_intrinsic_value_idx]  # shape: (N, 144), N = #rows
        right_camera_intrinsic = landmarks_input[:, self._right_camera_first_intrinsic_value_idx:self._first_right_2_left_matrix_value_idx]  # shape: (N, 9), N = #rows
        right_2_left_mat = landmarks_input[:, self._first_right_2_left_matrix_value_idx:]  # shape: (N, 16), N = #rows

        left_camera_lmks = left_camera_lmks.reshape(-1, 3, 48)  # shape: (N, 3, 48)
        right_camera_lmks = right_camera_lmks.reshape(-1, 3, 48)  # shape: (N, 3, 48)
        left_camera_lmks_z = left_camera_lmks[:, -1, :]  # shape: (N, 48)
        right_camera_lmks_z = right_camera_lmks[:, -1, :]  # shape: (N, 48)

        scaled_landmarks_input = np.concatenate([left_camera_lmks_z, 
            left_camera_intrinsic,
            right_camera_lmks_z,
            right_camera_intrinsic,
            right_2_left_mat], axis=1)  # shape: (N, 130)

        scaled_landmarks_input = self._minmax_scaler.transform(scaled_landmarks_input)
        scaled_landmarks_input = pd.DataFrame(scaled_landmarks_input, columns=self._columns_to_scale)
        scaled_left_camera_lmks_z = scaled_landmarks_input.loc[:, "left_shoulder_cam_left_z":"right_pinky_tip_cam_left_z"]  # shape: (N, 48)
        scaled_right_camera_lmks_z = scaled_landmarks_input.loc[:, "left_shoulder_cam_right_z":"right_pinky_tip_cam_right_z"]  # shape: (N, 48)
        scaled_left_camera_intr = scaled_landmarks_input.loc[:, "left_camera_intrinsic_x11":"left_camera_intrinsic_x33"]  # shape: (N, 9)
        scaled_right_camera_intr = scaled_landmarks_input.loc[:, "right_camera_intrinsic_x11":"right_camera_intrinsic_x33"]  # shape: (N , 9)
        scaled_right_2_left_mat = scaled_landmarks_input.loc[:, "right_to_left_matrix_x11":"right_to_left_matrix_x44"]  # shape: (N, 16)

        left_camera_lmks[:, -1, :] = scaled_left_camera_lmks_z  # shape: (N, 3, 48)
        right_camera_lmks[:, -1, :] = scaled_right_camera_lmks_z  # shape: (N, 3, 48)
        left_camera_lmks = left_camera_lmks.reshape(-1, 3 * 48)  # shape: (N, 144)
        right_camera_lmks = right_camera_lmks.reshape(-1, 3 * 48)  # shape:( N, 144)

        scaled_landmarks_input = np.concatenate([left_camera_lmks,
            scaled_left_camera_intr,
            right_camera_lmks,
            scaled_right_camera_intr,
            scaled_right_2_left_mat], axis=1)

        return scaled_landmarks_input