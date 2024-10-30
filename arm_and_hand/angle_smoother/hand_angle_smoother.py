from .angle_smoother import AngleSmoother
from .kalman_filter import KalmanFilter

class HandAngleSmoother(AngleSmoother):
    def __init__(self, angles_noise_statistical_file, dim):
        """
        Now, regarding to TomOSPC, he has 5 fingers, each finger 
        has 3 dof. For more details, checkout file:
                `.../configuration/{left/right}_hand_angles_stats.json`

        In this module, we suppose that a joint 1 of each finger is 
        a joint 1 of real robot (not from the `angle_calculator`).
        """

        super().__init__(angles_noise_statistical_file, dim)

        FINGERS_NAME = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]
        NUM_ANGLES_EACH_FINGER = 3
        NUM_HAND_ANGLES = NUM_ANGLES_EACH_FINGER * len(FINGERS_NAME)

        for angle_idx in range(NUM_HAND_ANGLES):
            finger_i = angle_idx // NUM_ANGLES_EACH_FINGER
            joint_of_finger_i = angle_idx % NUM_ANGLES_EACH_FINGER
            finger_i_stat = self._angles_stats[FINGERS_NAME[finger_i]]
            joint_i_stat = finger_i_stat[f"joint{joint_of_finger_i + 1}"]
            measure_noise = joint_i_stat["measure_noise"]
            init_cov = joint_i_stat["cov"]
            init_angle = joint_i_stat["init_angle"]

            f = KalmanFilter(
                dim=self._dim,
                measurement_noise=measure_noise,
                init_expectation=init_angle,
                init_cov=init_cov
            )

            self._kalman_filter_manager.append(f)