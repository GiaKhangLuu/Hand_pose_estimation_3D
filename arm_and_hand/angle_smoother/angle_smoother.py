import os
import json
import numpy as np

class AngleSmoother():
    def __init__(self, angles_noise_statistical_file, dim=1):
        """
        Currently, use Kalman Filter with `dim = 1` to smooth the angle 
        through frames. This means each angle is independent with each other.
        """
        with open(angles_noise_statistical_file, 'r') as f:
            self._angles_stats = json.load(f)

        assert dim == 1
        self._dim = dim

        self._kalman_filter_manager = []  

    def __call__(self, angles):
        """
        Input:
            angles (np.array): shape = (N,) for N angles
        Output:
            smoothed_angles (np.array): shape = (N,) for N angles
        """

        assert self._kalman_filter_manager is not None
        assert len(self._kalman_filter_manager) == angles.shape[0]

        smoothed_angles = np.zeros_like(angles, dtype=np.float32)
        for i, kalman_filter in enumerate(self._kalman_filter_manager):
            raw_angle = np.array([angles[i]])
            smoothed_angles[i] = kalman_filter(raw_angle).squeeze()

        return smoothed_angles