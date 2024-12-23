"""
Regarding to calculate three angles of index fingers to pinky finger, we need to follow the rule that a first joint rotates about
the z-axis of the parent coordinate and the second joint rotates about the y-axis of the first coordinate. Assuming that we 
want to calculate all three angles of the index finger (LF_joint_21 to LF_joint_23), we split this finger into two parts. The 
first joint of the first part rotates about the z-axis that is LF_joint_22, and the second joint of the first part rotates
about the y-axis that is LF_joint_21. Finally, the first joint of the second part rotates about the z-axis that is LF_joint_23.

To get the correct coordinates of a specified finger when it is at home position (0 degree for all angles), we need to rotate the 
last coordinate of the wrist to make sure it align with the finger coordinate when the finger is in home position. To get this,
we have to follow these steps. In this case, we want to find a home position of index finger

    1. Open URDF of robot in RVIZ, get a name of `wrist` link and a name of a first link of the index finger. Note that we have
        to ensure that all of these links are in home position (0 degree).
    2. To get the quaternion vector from wrist to index finger, we use the command:
        ```
        $ ros2 run tf2_ros tf2_echo [reference_frame] [target_frame]
        ```
        In this case the command is:
        ```
        $ ros2 run tf2_ros tf2_echo Left_arm_endlink LF_link_21
        ```
    3. The value we get from the above command is the quaternion vector formated: (x, y, z, w), Therefore, we have to 
        convert it to a rotation matrix. Let name this rotation matrix if `A`.
    4. Unfortunately, the current coordinate of the last joint of the wrist (joint 6) is not in the same direction as 
        the wrist link in RVIZ. Therefore, we have to rotate the current coordinate of the last joint to the same direction as
        the wrist link in RVIZ before multiplying with the matrix A. Let name this rotation matrix which rotates our current
        coordinate of the last joint to coordinate of the wrist in RVIZ is `B`. 
    5. Now our coordinate is in a correct position, but the first joint does not rotate about the z-axis and the second joint
        does not rotate about the y-axis. Therefore, we need to find a rotation matrix `C` to make sure our angle calculation
        function works correctly.

The entire formulation is:
    FINGER_HOME_COORD = J6_RM_WRT_0 @ B @ A @ C

Which:
    J6_RM_WRT_O: the coordinate of the last joint from an arm (joint 6)
    B: Rotation matrix to rotate the current joint6_coordinate to the same direction as the wrist of 
        robot in RVIZ (when it is in home position - 0 degree).
    A: Rotation matrix to align the wrist coordinate to the finger coordinate when it is in home position - 0 degree.
    C: Rotation matrix to ensure that the first joint rotates about the z-axis and the second joint rotates about the 
        y-axis.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from .chain_angle_calculator import ChainAngleCalculator

mcp_vector_in_init_frame = None
pip_vector_in_init_frame = None

rot_mat_to_rearrange_finger_coord = R.from_euler("x", 90, degrees=True).as_matrix()
rot_mat_for_pip = np.eye(3)

STATIC_BOUND = 1
joint1_min = -80
joint1_max = -1
joint2_min = -30
joint2_max = 30
joint3_min = -70
joint3_max = -1

class LeftFingerAngleCalculator(ChainAngleCalculator):
    def __init__(
        self, 
        num_chain, 
        finger_name, 
        landmark_dictionary, 
        last_coord_of_robot_to_home_position_of_finger_quat,
        last_coord_of_real_person_to_last_coord_of_robot_in_rviz_rot_mat
    ):
        """
        TODO: Doc.
        """
        self.finger_name = finger_name
        self.landmarks_name = ["MCP", "PIP"]
        self._STATIC_BOUND = STATIC_BOUND
        self._mapping_to_robot_angle_func_container = [
            (self._joint1_angle_to_TomOSPC_angle, self._joint2_angle_to_TomOSPC_angle), 
            (self._joint3_angle_to_TomOSPC_angle, None)
        ]
        self._vector_landmark_in_previous_frame_container = [
            mcp_vector_in_init_frame,
            pip_vector_in_init_frame
        ]
        # Just use when minimum is negative and maximum is positive
        self._angle_range_of_two_joints_container = [
            [[joint1_min + self._STATIC_BOUND, joint1_max - self._STATIC_BOUND], 
             [joint2_min + self._STATIC_BOUND, joint2_max - self._STATIC_BOUND]],
            [[joint3_min + self._STATIC_BOUND, joint3_max - self._STATIC_BOUND], 
             [None, None]],
        ]
        self._axis_to_get_the_opposite_if_angle_exceed_the_limit_of_two_joints_container = [
            [None, None],
            [None, None]
        ]
        self._get_the_opposite_of_two_joints_flag_container = [
            [False, False],
            [False, False] 
        ]
        self._limit_angle_of_two_joints_flag_container = [
            [False, False],
            [False, False]
        ]
        self.calculate_second_angle_flag_container = [True, False]
        self._clip_angle_of_two_joints_flag_container = [
            [True, True],
            [True, True]
        ]

        self._last_coord_of_real_person_to_last_coord_in_rviz_rot_mat = last_coord_of_real_person_to_last_coord_of_robot_in_rviz_rot_mat
        self._last_coord_of_robot_to_home_position_of_finger_rot_mat = R.from_quat(
            last_coord_of_robot_to_home_position_of_finger_quat).as_matrix()
        rot_mat_for_mcp = (self._last_coord_of_real_person_to_last_coord_in_rviz_rot_mat @ 
            self._last_coord_of_robot_to_home_position_of_finger_rot_mat @
            rot_mat_to_rearrange_finger_coord)
        self.rot_mat_to_rearrange_container = [rot_mat_for_mcp, rot_mat_for_pip]
        
        self._apply_dynamic_bound = True
        # Open RVIZ to get these values
        self._mcp_joint_1_angle_checkpoint = [joint1_max, -40, -48]
        self._mcp_max_joint_2_angle_adapted_for_joint_1 = [joint2_max, 28, 25]
        assert len(self._mcp_joint_1_angle_checkpoint) == len(self._mcp_max_joint_2_angle_adapted_for_joint_1)
        
        super().__init__(num_chain, landmark_dictionary)

    def _get_landmark_vector(self, chain_idx, XYZ_landmarks):
        landmark_name = self.landmarks_name[chain_idx]
        if self.finger_name in ["INDEX", "MIDDLE", "RING"]:
            finger_landmark_name = f"{self.finger_name}_FINGER_{landmark_name}"
        else:
            finger_landmark_name = f"{self.finger_name}_{landmark_name}"
        landmark_idx = self._landmark_dictionary.index(finger_landmark_name)
        next_landmark_idx = landmark_idx + 1
        landmark_vec_wrt_origin = XYZ_landmarks[landmark_idx].copy() 
        next_landmark_vec_wrt_origin = XYZ_landmarks[next_landmark_idx].copy()
        landmark_vec = next_landmark_vec_wrt_origin - landmark_vec_wrt_origin 
        return landmark_vec
    
    def _calculate_dynamic_limit_max(self, mcp_joint_1_angle):
        """
        Due to the mechanic design of TomO's hands, each finger mcp's joint 2
        has a different collision value which depends on the value of joint 1. 
        The more finger mcp's joint 1 moves up, the less space joint 2 is 
        able to move horizontally.
        Note that in the scope of this source code, finger mcp's joint 2 is the 
        finger mcp's joint 1 of the real robot and finger mcp's joint 1 is the
        finger mcp's joint 2 of the real robot. 
        """
        
        # np.interp needs to use increasing list, but our angle is in 
        # decreasing format => have to reverse them  
        dynamic_limit_max = np.interp(mcp_joint_1_angle, 
                                      self._mcp_joint_1_angle_checkpoint[::-1],
                                      self._mcp_max_joint_2_angle_adapted_for_joint_1[::-1])
        return dynamic_limit_max

    def _joint1_angle_to_TomOSPC_angle(self, joint1_angle):
        """
        TODO: doc. 
        Input: 
            joint1_angle (float):
        Output:
            tomospc_angle_j1 (int): 
        """
        tomospc_angle_j1 = joint1_angle 
        return tomospc_angle_j1 

    def _joint2_angle_to_TomOSPC_angle(self, joint2_angle):
        """
        TODO: doc. 
        Input: 
            joint2_angle (float):
        Output:
            tomospc_angle_j2 (int): 
        """
        tomospc_angle_j2 = -joint2_angle
        return tomospc_angle_j2 

    def _joint3_angle_to_TomOSPC_angle(self, joint3_angle):
        """
        TODO: doc. 
        Input: 
            joint3_angle (float):
        Output:
            tomospc_angle_j3 (int): 
        """
        tomospc_angle_j3 = joint3_angle 
        return tomospc_angle_j3 

    def __call__(self, XYZ_landmarks, parent_coordinate):
        merged_result_dict = dict()
        chain_result_dict = self._calculate_chain_angles(XYZ_landmarks, parent_coordinate)
        finger_angles = []
        finger_rot_mats_wrt_origin = []
        finger_rot_mats_wrt_parent = []

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
            
            if (self._apply_dynamic_bound and 
                self.calculate_second_angle_flag_container[chain_idx] and
                chain_idx == 0):
                dynamic_limit_max = self._calculate_dynamic_limit_max(older_brother_angle)
                younger_brother_angle = np.clip(younger_brother_angle, -dynamic_limit_max, dynamic_limit_max)                                                                       

            if self.calculate_second_angle_flag_container[chain_idx]:
                extended_angles = [older_brother_angle, younger_brother_angle]
                extended_rot_mats_wrt_origin = [older_brother_rot_mat_wrt_origin, younger_brother_rot_mat_wrt_origin]
                extended_rot_mats_wrt_parent = [older_brother_rot_mat_wrt_parent, younger_brother_rot_mat_wrt_older_brother]
            else:
                extended_angles = [older_brother_angle]
                extended_rot_mats_wrt_origin = [older_brother_rot_mat_wrt_origin]
                extended_rot_mats_wrt_parent = [older_brother_rot_mat_wrt_parent]

            finger_angles.extend(extended_angles)
            finger_rot_mats_wrt_origin.extend(extended_rot_mats_wrt_origin)
            finger_rot_mats_wrt_parent.extend(extended_rot_mats_wrt_parent)

        merged_result_dict[self.finger_name] = {
            "angles": finger_angles,
            "rot_mats_wrt_origin": finger_rot_mats_wrt_origin,
            "rot_mats_wrt_parent": finger_rot_mats_wrt_parent
        }

        return merged_result_dict