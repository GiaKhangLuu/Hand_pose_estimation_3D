import numpy as np
from scipy.spatial.transform import Rotation as R
from .arm_angle_calculator import ArmAngleCalculator

shoulder_vector_in_init_frame = None
elbow_vector_in_init_frame = None
wrist_vector_in_init_frame = None

rotation_matrix_for_shoulder = np.eye(3)
rotation_matrix_for_elbow = R.from_euler("xz", [90, -90], degrees=True).as_matrix()
rotation_matrix_for_wrist = R.from_euler("y", -90, degrees=True).as_matrix()

STATIC_BOUND = 2
joint1_min = -86
joint1_max = 195
joint2_min = -92
joint2_max = 3
joint3_min = -143
joint3_max = 143
joint4_min = -22
joint4_max = 91
joint5_min = -206
joint5_max = 115
joint6_min = -69
joint6_max = 52

class RightArmAngleCalculator(ArmAngleCalculator):
    def __init__(self, num_chain, landmark_dictionary):
        """
        TODO: Doc.
        """
        self.landmarks_name = ("right shoulder", "right elbow", "RIGHT_WRIST")  # keep a capital word in order for compatibility with the `landmark_dictionary`
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
        self._STATIC_BOUND = STATIC_BOUND
        # Just use when minimum is negative and maximum is positive
        self._angle_range_of_two_joints_container = [
            [[joint1_min + self._STATIC_BOUND, joint1_max - self._STATIC_BOUND], 
             [joint2_min + self._STATIC_BOUND, joint2_max - self._STATIC_BOUND]],
            [[joint3_min + self._STATIC_BOUND, joint3_max - self._STATIC_BOUND], 
             [joint4_min + self._STATIC_BOUND, joint4_max - self._STATIC_BOUND]],
            [[joint5_min + self._STATIC_BOUND, joint5_max - self._STATIC_BOUND], 
             [joint6_min + self._STATIC_BOUND, joint6_max - self._STATIC_BOUND]]
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
            [True, False]
        ]

        super().__init__(num_chain, landmark_dictionary, "right")
        
    def _joint1_angle_to_TomOSPC_angle(self, joint1_angle):
        """
        TODO: doc. 
        Input: 
            angle (float):
        Output:
            tomospc_angle_j1 (int): 
        """
        if -90 <= joint1_angle <= 180:
            tomospc_angle_j1 = joint1_angle
        else:
            tomospc_angle_j1 = 360 + joint1_angle
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
        tomospc_angle_j4 = -joint4_angle
        return tomospc_angle_j4

    def _joint5_angle_to_TomOSPC_angle(self, joint5_angle):
        """
        TODO: doc. 
        Input: 
            angle (float):
        Output:
            tomospc_angle_j5 (int): 
        """
        global joint5_min
        if -115 <= joint5_angle <= 180:
            tomospc_angle_j5 = -joint5_angle
        else:
            tomospc_angle_j5 = min(joint5_min, -(joint5_angle - 360))
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
        if landmark_name == "right shoulder":
            landmark_vec = self._get_shoulder_vector(XYZ_landmarks)
        elif landmark_name == "right elbow":
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
        right_shoulder_landmark = XYZ_landmarks[self._landmark_dictionary.index("right shoulder")].copy() 
        right_elbow_landmark = XYZ_landmarks[self._landmark_dictionary.index("right elbow")].copy()
        shoulder_vec = right_elbow_landmark - right_shoulder_landmark
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
        right_wrist_landmark = XYZ_landmarks[self._landmark_dictionary.index("RIGHT_WRIST")].copy()
        right_elbow_landmark = XYZ_landmarks[self._landmark_dictionary.index("right elbow")].copy()
        elbow_vec = right_wrist_landmark - right_elbow_landmark
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
        right_wrist_landmark = XYZ_landmarks[self._landmark_dictionary.index("RIGHT_WRIST")].copy()
        right_index_finger_landmark = XYZ_landmarks[self._landmark_dictionary.index("RIGHT_INDEX_FINGER_MCP")].copy()
        right_middle_finger_landmark = XYZ_landmarks[self._landmark_dictionary.index("RIGHT_MIDDLE_FINGER_MCP")].copy()

        right_u_wrist = right_index_finger_landmark - right_wrist_landmark
        right_v_wrist = right_middle_finger_landmark - right_wrist_landmark

        wrist_vec = np.cross(right_u_wrist, right_v_wrist)
        return wrist_vec
    