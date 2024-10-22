import sklearn
import joblib
import numpy as np
import pandas as pd

class LandmarksScaler():
    """
    Currently, use MinMaxScaler from sklearn to scale XYZ 
    """

    def __init__(self, scaler_path=None):
        assert scaler_path is not None 
        self.minmax_scaler = joblib.load(scaler_path)
        assert isinstance(self.minmax_scaler, sklearn.preprocessing._data.MinMaxScaler)

    def __call__(self, landmarks_input):
        """
        Input:
            landmarks_input (np.array): landmarks input to scale, shape = (N, self._num_features), N = #data
        Output:
            scaled_landmarks_input (np.array): shape = (N, self._num_features)
        """
        assert landmarks_input.ndim == 2
        scaled_landmarks_input = self.minmax_scaler.transform(landmarks_input)
        return scaled_landmarks_input