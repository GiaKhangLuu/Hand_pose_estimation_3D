import numpy as np
from scipy.spatial.transform import Rotation as R
from chain_angle_calculator import ChainAngleCalculator

shoulder_vector_in_init_frame = None
elbow_vector_in_init_frame = None
wrist_vector_in_init_frame = None

rotation_matrix_for_shoulder = np.eye(3)
rotation_matrix_for_elbow = R.from_euler("xz", [90, -90], degrees=True).as_matrix()
rotation_matrix_for_wrist = R.from_euler("y", -90, degrees=True).as_matrix()

bound = 2
joint1_min = -195 + bound
joint1_max = 86 - bound
joint2_min = -3 + bound
joint2_max = 92 - bound
joint3_min = -143 + bound
joint3_max = 143 - bound
joint4_min = -91 + bound
joint4_max = 22 - bound
joint5_min = -115 + bound
joint5_max = 206 - bound
joint6_min = -69 + bound
joint6_max = 52 - bound

class LeftArmAngleCalculator(ChainAngleCalculator):
    def __init__(self, num_chain, landmark_dictionary):
        self.landmarks_name = ("shoulder", "elbow", "WRIST")  # keep a capital word in order for compatibility with the `landmark_dictionary`
        self._mapping_to_robot_angle_func_container = [
            (self._joint1_angle_to_TomOSPC_angle, self._joint2_angle_to_TomOSPC_angle),
            (self._joint3_angle_to_TomOSPC_angle, self._joint4_angle_to_TomOSPC_angle),
            (self._joint5_angle_to_TomOSPC_angle, self._joint6_angle_to_TomOSPC_angle)
        ]
        self._vector_landmark_in_previous_frame_container = [
            shoulder_vector_in_init_frame,
            elbow_vector_in_init_frame,
            wrist_vector_in_init_frame
        ]
        self.rot_mat_to_rearrange_container = [
            rotation_matrix_for_shoulder,
            rotation_matrix_for_elbow,
            rotation_matrix_for_wrist
        ]
        self._angle_range_of_two_joints_container = [
            [[joint1_min, joint1_max], [joint2_min, joint2_max]],
            [[joint3_min, joint3_max], [joint4_min, joint4_max]],
            [[joint5_min, joint5_max], [joint6_min, joint6_max]]
        ]
        self._axis_to_get_the_opposite_if_angle_exceed_the_limit_of_two_joints_container = [
            ["y", None],
            [None, None], 
            [None, None]
        ]
        self._get_the_opposite_of_two_joints_flag_container = [
            [True, True],
            [True, True],
            [True, True]
        ]
        self._limit_angle_of_two_joints_flag_container = [
            [True, False],
            [False, False],
            [False, False]
        ]
        self.calculate_second_angle_flag_container = [True, True, True]
        self._clip_angle_of_two_joints_flag_container = [
            [True, True],
            [True, True],
            [True, True]
        ]

        super().__init__(num_chain, landmark_dictionary)

    def _joint1_angle_to_TomOSPC_angle(self, joint1_angle):
        """
        TODO: doc. 
        Input: 
            angle (float):
        Output:
            tomospc_angle_j1 (int): 
        """
        if -90 <= joint1_angle <= 180:
            tomospc_angle_j1 = -joint1_angle
        else:
            tomospc_angle_j1 = -joint1_angle - 360
        return tomospc_angle_j1 

    def _joint2_angle_to_TomOSPC_angle(self, joint2_angle):
        """
        TODO: doc. 
        Input: 
            angle (float):
        Output:
            tomospc_angle_j2 (int): 
        """
        tomospc_angle_j2 = -joint2_angle
        return tomospc_angle_j2

    def _joint3_angle_to_TomOSPC_angle(self, joint3_angle):
        """
        TODO: doc. 
        Input: 
            angle (float):
        Output:
            tomospc_angle_j3 (int): 
        """
        tomospc_angle_j3 = -joint3_angle
        return tomospc_angle_j3

    def _joint4_angle_to_TomOSPC_angle(self, joint4_angle):
        """
        TODO: doc. 
        Input: 
            angle (float):
        Output:
            tomospc_angle_j4 (int): 
        """
        tomospc_angle_j4 = joint4_angle
        return tomospc_angle_j4

    def _joint5_angle_to_TomOSPC_angle(self, joint5_angle):
        """
        TODO: doc. 
        Input: 
            angle (float):
        Output:
            tomospc_angle_j5 (int): 
        """
        global joint5_max
        if joint5_angle > 115:
            tomospc_angle_j5 = min(joint5_max, -joint5_angle + 360)
        else:
            tomospc_angle_j5 = -joint5_angle

        return tomospc_angle_j5

    def _joint6_angle_to_TomOSPC_angle(self, joint6_angle):
        """
        TODO: doc. 
        Input: 
            angle (float):
        Output:
            tomospc_angle_j6 (int): 
        """
        tomospc_angle_j6 = -joint6_angle
        return tomospc_angle_j6

    def _get_landmark_vector(self, chain_idx, XYZ_landmarks):
        landmark_name = self.landmarks_name[chain_idx]
        if landmark_name == "shoulder":
            landmark_vec = self._get_shoulder_vector(XYZ_landmarks)
        elif landmark_name == "elbow":
            landmark_vec = self._get_elbow_vector(XYZ_landmarks)
        else:
            landmark_vec = self._get_wrist_vector(XYZ_landmarks)
        return landmark_vec

    def _get_shoulder_vector(self, XYZ_landmarks):
        """
        TODO: Doc.
        Get shoulder vector to compute angle j1 and angle j2
        Input:
            XYZ_landmarks:
        Output:
            shoulder_vec:
        """
        shoulder_vec = XYZ_landmarks[self._landmark_dictionary.index("left elbow")].copy()
        return shoulder_vec

    def _get_elbow_vector(self, XYZ_landmarks):
        """
        TODO: Doc.
        Get elbow vector to compute angle j3 and angle j4
        Input:
            XYZ_landmarks:
        Output:
            elbow_vec:
        """
        wrist_landmark = XYZ_landmarks[self._landmark_dictionary.index("WRIST")].copy()
        left_elbow_landmark = XYZ_landmarks[self._landmark_dictionary.index("left elbow")].copy()
        elbow_vec = wrist_landmark - left_elbow_landmark
        return elbow_vec

    def _get_wrist_vector(self, XYZ_landmarks):
        """
        TODO: Doc.
        Get wrist vector to compute angle j5 and angle j6
        Input:
            XYZ_landmarks:
        Output:
            wrist_vec:
        """
        wrist_landmark = XYZ_landmarks[self._landmark_dictionary.index("WRIST")].copy()
        index_finger_landmark = XYZ_landmarks[self._landmark_dictionary.index("INDEX_FINGER_MCP")].copy()
        middle_finger_landmark = XYZ_landmarks[self._landmark_dictionary.index("MIDDLE_FINGER_MCP")].copy()

        u_wrist = index_finger_landmark - wrist_landmark
        v_wrist = middle_finger_landmark - wrist_landmark

        wrist_vec = np.cross(v_wrist, u_wrist)
        return wrist_vec

    def __call__(self, XYZ_landmarks, parent_coordinate):
        merged_result_dict = dict()
        chain_result_dict = self._calculate_chain_angles(XYZ_landmarks, parent_coordinate)
        left_arm_angles = []
        left_arm_rot_mats_wrt_origin = []
        left_arm_rot_mats_wrt_parent = []

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

            left_arm_angles.extend([older_brother_angle, younger_brother_angle])
            left_arm_rot_mats_wrt_origin.extend([older_brother_rot_mat_wrt_origin, younger_brother_rot_mat_wrt_origin])
            left_arm_rot_mats_wrt_parent.extend([older_brother_rot_mat_wrt_parent, younger_brother_rot_mat_wrt_older_brother])

        merged_result_dict["left_arm"] = {
            "angles": left_arm_angles,
            "rot_mats_wrt_origin": left_arm_rot_mats_wrt_origin,
            "rot_mats_wrt_parent": left_arm_rot_mats_wrt_parent
        }

        return merged_result_dict