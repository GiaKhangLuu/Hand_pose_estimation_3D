import numpy as np
from scipy.spatial.transform import Rotation as R
from .chain_angle_calculator import ChainAngleCalculator

head_vector_in_init_frame = None

rotation_matrix_for_eye = R.from_euler("xz", [90, 90], degrees=True).as_matrix()

STATIC_BOUND = 1
joint1_min = -57 
joint1_max = 57
joint2_min = -6
joint2_max = 66

class HeadAngleCalculator(ChainAngleCalculator):
    def __init__(self, num_chain, landmark_dictionary):
        self.landmarks_name = ("eye",)
        self._mapping_to_robot_angle_func_container = [
            (self._joint1_angle_to_TomOSPC_angle, self._joint2_angle_to_TomOSPC_angle)
        ]
        self._vector_landmark_in_previous_frame_container = [
            head_vector_in_init_frame,
        ]
        self.rot_mat_to_rearrange_container = [
            rotation_matrix_for_eye,
        ]
        self._STATIC_BOUND = STATIC_BOUND

        # Just use when minimum is negative and maximum is positive
        self._angle_range_of_two_joints_container = [
            [[joint1_min + self._STATIC_BOUND, joint1_max - self._STATIC_BOUND], 
             [joint2_min + self._STATIC_BOUND, joint2_max - self._STATIC_BOUND]]
        ]

        self._axis_to_get_the_opposite_if_angle_exceed_the_limit_of_two_joints_container = [
            [None, None],
        ]
        self._get_the_opposite_of_two_joints_flag_container = [
            [False, False],
        ]
        self._limit_angle_of_two_joints_flag_container = [
            [False, False],
        ]
        self.calculate_second_angle_flag_container = [True]
        self._clip_angle_of_two_joints_flag_container = [
            [True, False],
        ]

        super().__init__(num_chain, landmark_dictionary)

    def _joint1_angle_to_TomOSPC_angle(self, joint1_angle):
        """
        TODO: Doc.
        Input:
            angle (float):
        Output:
            tomospc_angle_j1 (float):
        """
        tomospc_angle_j1 = -joint1_angle
        return tomospc_angle_j1

    def _joint2_angle_to_TomOSPC_angle(self, joint2_angle):
        """
        TODO: Doc.
        Input:
            angle (float):
        Output:
            tomospc_angle_j2 (float):
        """
        tomospc_angle_j2 = -joint2_angle
        return tomospc_angle_j2

    def _get_landmark_vector(self, chain_idx, XYZ_landmarks):
        landmark_name = self.landmarks_name[chain_idx]
        right_eye = XYZ_landmarks[self._landmark_dictionary.index(f"right {landmark_name}")].copy()
        left_eye = XYZ_landmarks[self._landmark_dictionary.index(f"left {landmark_name}")].copy()
        nose = XYZ_landmarks[self._landmark_dictionary.index("nose")].copy()

        left_vec = nose - left_eye
        right_vec = nose - right_eye

        head_vec = np.cross(left_vec, right_vec)

        return head_vec

    def __call__(self, XYZ_landmarks, parent_coordinate):
        merged_result_dict = dict()
        chain_result_dict = self._calculate_chain_angles(XYZ_landmarks, parent_coordinate)
        angles = []
        rot_mats_wrt_origin = []
        rot_mats_wrt_parent = []

        for chain_idx in range(self.num_chain):
            chain_result = chain_result_dict[f"chain_{chain_idx+1}"] 

            older_brother_angle = chain_result["older_brother_angle"]
            older_brother_rot_mat_wrt_origin = chain_result["older_brother_rot_mat_wrt_origin"]
            older_brother_rot_mat_wrt_parent = chain_result["older_brother_rot_mat_wrt_parent"]
            younger_brother_angle = chain_result["younger_brother_angle"]
            younger_brother_rot_mat_wrt_origin = chain_result["younger_brother_rot_mat_wrt_origin"]
            younger_brother_rot_mat_wrt_older_brother = chain_result["younger_brother_rot_mat_wrt_older_brother"]
            vector_in_current_frame = chain_result["vector_in_current_frame"].copy()

            self._update_vector_in_previous_frame(chain_idx, vector_in_current_frame)

            angles.extend([older_brother_angle, younger_brother_angle])
            rot_mats_wrt_origin.extend([older_brother_rot_mat_wrt_origin, younger_brother_rot_mat_wrt_origin])
            rot_mats_wrt_parent.extend([older_brother_rot_mat_wrt_parent, younger_brother_rot_mat_wrt_older_brother])

        merged_result_dict["head"] = {
            "angles": angles,
            "rot_mats_wrt_origin": rot_mats_wrt_origin,
            "rot_mats_wrt_parent": rot_mats_wrt_parent
        }

        return merged_result_dict