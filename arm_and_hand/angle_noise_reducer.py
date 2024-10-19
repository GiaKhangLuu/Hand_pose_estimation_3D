import os
from pathlib import Path
import json
import numpy as np
from kalman_filter import KalmanFilter

class AngleNoiseReducer():
    def __init__(self, angles_noise_statistical_file, num_angles=6, dim=1):
        """
        Attempt to reduce noises of angles  when arm does not move.
        """
        with open(angles_noise_statistical_file, 'r') as file:
            angles_stats = json.load(file)

        assert dim in [1, 6]
        self._dim = dim
        self._kalman_filters = []
        if self._dim == 1:
            self._joints_name = []
            for i in range(num_angles):
                self._joints_name.append("joint{}".format(i + 1))

            for j_name in self._joints_name:
                joint_stat = angles_stats[j_name]
                measure_noise = joint_stat["measure_noise"]
                initial_cov = joint_stat["cov"]
                initial_est = joint_stat["init_angle"]

                f = KalmanFilter(dim=self._dim, 
                    measurement_noise=measure_noise, 
                    init_expectation=initial_est,
                    init_cov=initial_cov)

                self._kalman_filters.append(f)
        else:
            measure_noise = np.array(angles_stats["measure_noise"], dtype=np.float64).reshape(6, 6)
            initial_cov = np.array(angles_stats["cov"], dtype=np.float64).reshape(6, 6)
            initial_est = np.array(angles_stats["init_angle"], dtype=np.float64)

            self._kalman_filters = KalmanFilter(dim=self._dim,
                measurement_noise=measure_noise,
                init_expectation=initial_est,
                init_cov=initial_cov)

    def __call__(self, angles):
        """
        Input:
            angles (np.array): shape = (6,) for 6 arm angles
        Output:
            smoothed_angles (np.array): shape = (6,) for 6 arm angles
        """
        smoothed_angles = np.zeros_like(angles)
        if self._dim == 1:
            for i in range(len(self._joints_name)):
                raw_angle = np.array([angles[i]])
                kalman_filter = self._kalman_filters[i]
                smoothed_angles[i] = kalman_filter(raw_angle).squeeze()
        else:
            smoothed_angles = self._kalman_filters(angles)

        return smoothed_angles
