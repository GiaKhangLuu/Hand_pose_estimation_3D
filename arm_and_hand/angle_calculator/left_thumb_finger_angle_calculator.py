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
from .left_finger_angle_calculator import LeftFingerAngleCalculator

mcp_vector_in_init_frame = None
ip_vector_in_init_frame = None

rot_mat_to_rearrange_finger_coord = np.eye(3)
rot_mat_for_ip = R.from_euler("x", 90, degrees=True).as_matrix()

STATIC_BOUND = 1
joint1_min = -74 
joint1_max = 45 
joint2_min = -90 
joint2_max = 0 
joint3_min = -90 
joint3_max = 0 

class LeftThumbFingerAngleCalculator(LeftFingerAngleCalculator):
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
        
        super().__init__(
            num_chain=num_chain, 
            finger_name=finger_name,
            landmark_dictionary=landmark_dictionary,
            last_coord_of_robot_to_home_position_of_finger_quat=last_coord_of_robot_to_home_position_of_finger_quat,
            last_coord_of_real_person_to_last_coord_of_robot_in_rviz_rot_mat=last_coord_of_real_person_to_last_coord_of_robot_in_rviz_rot_mat
        )

        self.landmarks_name = ["MCP", "IP"]
        self._mapping_to_robot_angle_func_container = [
            (self._joint1_angle_to_TomOSPC_angle, self._joint2_angle_to_TomOSPC_angle), 
            (self._joint3_angle_to_TomOSPC_angle, None)
        ]
        self._vector_landmark_in_previous_frame_container = [
            mcp_vector_in_init_frame,
            ip_vector_in_init_frame
        ]
        self._STATIC_BOUND = STATIC_BOUND
        # Just use when minimun is negative and maximum is positive
        self._angle_range_of_two_joints_container = [
            [[joint1_min + self._STATIC_BOUND, joint1_max - self._STATIC_BOUND], 
             [joint2_min + self._STATIC_BOUND, joint2_max - self._STATIC_BOUND]],
            [[joint3_min + self._STATIC_BOUND, joint3_max - self._STATIC_BOUND], 
             [None, None]],
        ]

        rot_mat_for_mcp = (self._last_coord_of_real_person_to_last_coord_in_rviz_rot_mat @ 
            self._last_coord_of_robot_to_home_position_of_finger_rot_mat @
            rot_mat_to_rearrange_finger_coord)
        self.rot_mat_to_rearrange_container = [rot_mat_for_mcp, rot_mat_for_ip]
        
        self._apply_dynamic_bound = False

    def _joint1_angle_to_TomOSPC_angle(self, joint1_angle):
        """
        TODO: doc. 
        Input: 
            joint1_angle (float):
        Output:
            tomospc_angle_j1 (int): 
        """
        tomospc_angle_j1 = -joint1_angle 
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

    def _get_landmark_vector(self, chain_idx, XYZ_landmarks):
        landmark_name = self.landmarks_name[chain_idx]
        finger_landmark_name = f"{self.finger_name}_{landmark_name}"
        landmark_idx = self._landmark_dictionary.index(finger_landmark_name)
        next_landmark_idx = landmark_idx + 1
        landmark_vec_wrt_origin = XYZ_landmarks[landmark_idx].copy() 
        next_landmark_vec_wrt_origin = XYZ_landmarks[next_landmark_idx].copy()
        landmark_vec = next_landmark_vec_wrt_origin - landmark_vec_wrt_origin 
        return landmark_vec