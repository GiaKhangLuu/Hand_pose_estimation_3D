from .angle_smoother import AngleSmoother
from .kalman_filter import KalmanFilter

class ArmAngleSmoother(AngleSmoother):
    def __init__(self, angles_noise_statistical_file, dim):
        """
        Now, regarding to TomOSPC, an arm has 6 dof. For more
        information, checkout file:
            `../configuration/{left/right}_arm_angles_stats.json`
        """

        super().__init__(angles_noise_statistical_file, dim)

        for i in range(len(self._angles_stats.keys())):
            joint_stat = self._angles_stats[f"joint{i + 1}"]
            measured_noise = joint_stat["measure_noise"]
            init_cov = joint_stat["cov"]
            init_angle = joint_stat["init_angle"]

            f = KalmanFilter(
                dim=self._dim,
                measurement_noise=measured_noise,
                init_expectation=init_angle,
                init_cov=init_cov
            )

            self._kalman_filter_manager.append(f)