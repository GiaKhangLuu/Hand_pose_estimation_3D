import numpy as np

A = np.array([[1]])
Q = np.array([[1e-4]])  # for now, tunning this value when testing on real robot
G = np.array([[1]])
R = np.array([[0.01]])

def predict_and_correct(measured_angle, previous_angle, previous_sigma):
    # Prediction
    angle_prediction = A @ previous_angle
    sigma_prediction = A @ previous_sigma @ A.T + Q

    # Measurement
    kalman_gain = sigma_prediction @ G.T @ np.linalg.inv(G @ sigma_prediction @ G.T + R)
    angle_correction = angle_prediction + kalman_gain @ (measured_angle - G @ angle_prediction)
    sigma_correction = (np.eye(len(sigma_prediction)) - kalman_gain @ G) @ sigma_prediction

    return angle_correction, sigma_correction