import numpy as np

class KalmanFilter():
    def __init__(self, dim, measurement_noise, init_expectation, init_cov):
        self._dim = dim
        self._A = np.eye(self._dim)
        self._Q = np.eye(self._dim)
        self._G = np.eye(self._dim)

        if self._dim == 1:
            self._R = np.array([measurement_noise])
            self._expectation = np.array([init_expectation])
            self._cov = np.array([init_cov])
        else:
            self._R = measurement_noise
            self._expectation = init_expectation
            self._cov = init_cov

    def __call__(self, measurement):
        # Predict
        self._expectation = self._A @ self._expectation
        self._cov = self._A @ self._cov @ self._A.T + self._Q

        # Measurement
        kalman_gain = self._cov @ self._G.T @ np.linalg.inv(self._G @ self._cov @ self._G.T + self._R)

        expectation_correction = self._expectation + kalman_gain @ (measurement - self._G @ self._expectation)
        cov_correction = (np.eye(len(self._cov)) - kalman_gain @ self._G) @ self._cov

        self._expectation = expectation_correction
        self._cov = cov_correction

        return expectation_correction 