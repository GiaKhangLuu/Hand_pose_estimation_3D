import os
from pathlib import Path
import json
import numpy as np
from kalman_filter import KalmanFilter

class LandmarksNoiseReducer():
    def __init__(self, landmarks_noise_statistical_file):
        """
        Attempt to reduce noises of landmarks when arm does not move.
        """
        self._arm_hand_fused_names = ["left shoulder", "left elbow", "left hip", "right shoulder", "right hip", 
            "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP", "INDEX_FINGER_MCP", 
            "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP", "MIDDLE_FINGER_MCP", 
            "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP", "RING_FINGER_MCP", 
            "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP", "PINKY_MCP", "PINKY_PIP", 
            "PINKY_DIP", "PINKY_TIP", "right elbow"]

        with open(landmarks_noise_statistical_file, 'r') as file:
            landmarks_stats = json.load(file)

        self._kalman_filters = []
        for landmark_name in self._arm_hand_fused_names:
            landmark_stats = landmarks_stats[landmark_name]
            measure_noise = np.array(landmark_stats["measure_noise"]).reshape(3, 3)
            initial_cov = np.array(landmark_stats["cov"]).reshape(3, 3)
            initial_est = np.array(landmark_stats["filter_state_estimate"]) 

            f = KalmanFilter(dim=3, 
                measurement_noise=measure_noise, 
                init_expectation=initial_est,
                init_cov=initial_cov)

            self._kalman_filters.append(f)

    def __call__(self, landmarks):
        assert landmarks.shape == (len(self._arm_hand_fused_names), 3)
        filtered_landmarks = np.zeros_like(landmarks)
        for i in range(landmarks.shape[0]):
            raw_landmark = landmarks[i]
            kalman_filter = self._kalman_filters[i]
            filtered_landmarks[i] = kalman_filter(raw_landmark)

        return filtered_landmarks
