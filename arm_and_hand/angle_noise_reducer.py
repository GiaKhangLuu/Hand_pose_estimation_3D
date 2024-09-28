import os
from pathlib import Path
import json
import numpy as np
from kalman_filter import KalmanFilter

class AngleNoiseReducer():
    def __init__(self, angles_noise_statistical_file, num_angles=6):
        """
        Attempt to reduce noises of angles  when arm does not move.
        """
        with open(angles_noise_statistical_file, 'r') as file:
            angles_stats = json.load(file)

        self._joints_name = []
        for i in range(num_angles):
            self._joints_name.append("joint{}".format(i + 1))

        self._kalman_filters = []
        for j_name in self._joints_name:
            angle_stats = angles_stats[j_name]
            measure_noise = angle_stats["measure_noise"]
            initial_cov = angle_stats["cov"]
            initial_est = angle_stats["filter_state_estimate"]

            f = KalmanFilter(dim=1, 
                measurement_noise=measure_noise, 
                init_expectation=initial_est,
                init_cov=initial_cov)

            self._kalman_filters.append(f)

    def __call__(self, angles):
        """
        Input:
            angles (np.array): shape = (6,) for 6 arm angles
        Output:
            smoothed_angles (np.array): shape = (6,) for 6 arm angles
        """
        smoothed_angles = np.zeros_like(angles)
        for i in range(len(self._joints_name)):
            raw_angle = np.array([angles[i]])
            kalman_filter = self._kalman_filters[i]
            smoothed_angles[i] = kalman_filter(raw_angle).squeeze()

        return smoothed_angles
