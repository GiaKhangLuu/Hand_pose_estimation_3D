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
from angle_calculation_utilities import calculate_the_next_two_joints_angle

# These four quaternion vectors get from ROS2, which are formatted: (x, y, z, w)
wrist_to_home_position_index_finger_quat = [0.595, 0.517, 0.465, 0.404]
wrist_to_home_position_middle_finger_quat = [0.534, 0.508, 0.489, 0.466]
wrist_to_home_position_ring_finger_quat = [0.466, 0.489, 0.508, 0.534]
wrist_to_home_position_pinky_finger_quat = [0.404, 0.465, 0.517, 0.595]

j6_of_real_people_to_j6_in_rviz_rot_mat = R.from_euler("xz", [90, 180], degrees=True).as_matrix()
rot_mat_to_rearrange_finger_coord = R.from_euler("x", 90, degrees=True).as_matrix()
rot_mat_for_fingers_j3 = np.eye(3)

quaternion_vectors = [
    wrist_to_home_position_index_finger_quat,
    wrist_to_home_position_middle_finger_quat,
    wrist_to_home_position_ring_finger_quat,
    wrist_to_home_position_pinky_finger_quat]

bound = 2
joint1_max = 0 - bound
joint1_min = -90 + bound
joint2_max = 22 - bound
joint2_min = -22 + bound
joint3_max = 0 - bound
joint3_min = -90 + bound

class FingerAngles():
    def __init__(self, landmark_dictionary):
        """
        TODO: Doc.
        """
        self.fingers_name = ["INDEX", "MIDDLE", "RING", "PINKY"]
        self._rot_mat_to_cvt_last_coord_of_real_peoples_arm_to_last_coordinate_of_arm_in_rviz = j6_of_real_people_to_j6_in_rviz_rot_mat 
        self._rot_mat_to_rearrange_finger_coord = rot_mat_to_rearrange_finger_coord
        self._landmark_dictionary = landmark_dictionary
        self._rot_mat_to_align_last_coord_of_arm_to_finger_coords = self._cvt_quaternion_to_rot_mat(quaternion_vectors)
        self._map_real_person_angle_to_robot_angle_functions = (self._mapping_first_joint_angle, 
            self._mapping_second_joint_angle)
        self._previous_landmark_vectors_dict = self._init_previous_landmark_vectors()
        self._angles_ranges = [[joint1_min, joint1_max], 
            [joint2_min, joint2_max],
            [joint3_min, joint3_max],
            [None, None]]
        self._rot_mat_for_fingers_j3 = rot_mat_for_fingers_j3

    def _mapping_first_joint_angle(self, angle):
        return angle

    def _mapping_second_joint_angle(self, angle):
        return -angle

    def _init_previous_landmark_vectors(self):
        previous_landmark_vectors_dict = dict()
        for i, finger_name in enumerate(self.fingers_name):
            prev_vector = {
                "pre_pip_vec": None,
                "pre_dip_vec": None
            }
            previous_landmark_vectors_dict[finger_name] = prev_vector
        return previous_landmark_vectors_dict

    def _cvt_quaternion_to_rot_mat(self, quaternion_vectors):
        """
        TODO: Doc.
        """
        rot_mat_to_align_last_coord_of_arm_to_finger_coords = []
        for i, finger_name in enumerate(self.fingers_name):
            wrist_to_home_position_finger_i_quat = quaternion_vectors[i]
            wrist_to_home_position_finger_i_rot_mat = R.from_quat(wrist_to_home_position_finger_i_quat).as_matrix()
            rot_mat_to_align_last_coord_of_arm_to_finger_coords.append(wrist_to_home_position_finger_i_rot_mat)

        return rot_mat_to_align_last_coord_of_arm_to_finger_coords

    def _get_finger_landmark_vectors(self, XYZ_landmarks, finger_name):
        """ 
        TODO: Doc.
        """
        if finger_name in ["INDEX", "MIDDLE", "RING"]:
            finger_name = f"{finger_name}_FINGER"
        mcp_name = f"{finger_name}_MCP"
        pip_name = f"{finger_name}_PIP"
        dip_name = f"{finger_name}_DIP"

        mcp_landmark = XYZ_landmarks[self._landmark_dictionary.index(mcp_name)].copy()
        pip_landmark = XYZ_landmarks[self._landmark_dictionary.index(pip_name)].copy()
        dip_landmark = XYZ_landmarks[self._landmark_dictionary.index(dip_name)].copy()

        pip_vec = pip_landmark - mcp_landmark
        dip_vec = dip_landmark - pip_landmark

        return (pip_vec, dip_vec)

    def __call__(self, XYZ_landmarks, last_coordinate_of_arm):
        lefthand_angle_rs = dict()

        for i, finger_name in enumerate(self.fingers_name):
            pip_vec, dip_vec = self._get_finger_landmark_vectors(XYZ_landmarks, finger_name)
            pre_pip_vec = self._previous_landmark_vectors_dict[finger_name]["pre_pip_vec"]
            pre_dip_vec = self._previous_landmark_vectors_dict[finger_name]["pre_dip_vec"]

            rot_mat_to_align_to_home = self._rot_mat_to_align_last_coord_of_arm_to_finger_coords[i] 
            rot_mat_from_wrist_to_finger_i = (self._rot_mat_to_cvt_last_coord_of_real_peoples_arm_to_last_coordinate_of_arm_in_rviz 
                @ rot_mat_to_align_to_home @ self._rot_mat_to_rearrange_finger_coord)

            j1_j2_results = calculate_the_next_two_joints_angle(
                vector_landmark=pip_vec,
                map_to_robot_angle_funcs=self._map_real_person_angle_to_robot_angle_functions,
                parent_coordinate=last_coordinate_of_arm,
                vector_in_prev_frame=pre_pip_vec,
                rotation_matrix_to_rearrange_coordinate=rot_mat_from_wrist_to_finger_i,
                angle_range_of_two_joints=self._angles_ranges[:2],
                axis_to_get_the_opposite_if_angle_exceed_the_limit_of_two_joints=[None, None],
                get_the_opposite_of_two_joints=[False, False],
                limit_angle_of_two_joints=[False, False] 
            )

            finger_i_angle_j1 = j1_j2_results["older_brother_angle"]
            finger_i_j1_rot_mat_wrt_origin = j1_j2_results["older_brother_rot_mat_wrt_origin"]
            finger_i_j1_rot_mat_wrt_last_coordinate_of_arm = j1_j2_results["older_brother_rot_mat_wrt_parent"]
            finger_i_angle_j2 = j1_j2_results["younger_brother_angle"]
            finger_i_j2_rot_mat_wrt_origin = j1_j2_results["younger_brother_rot_mat_wrt_origin"]
            finger_i_j2_rot_mat_wrt_j1 = j1_j2_results["younger_brother_rot_mat_wrt_older_brother"] 
            finger_i_current_pip_vec = j1_j2_results["vector_in_current_frame"].copy()

            self._previous_landmark_vectors_dict[finger_name]["pre_pip_vec"] = finger_i_current_pip_vec 

            j3_j4_results = calculate_the_next_two_joints_angle(
                vector_landmark=dip_vec,
                map_to_robot_angle_funcs=self._map_real_person_angle_to_robot_angle_functions,
                parent_coordinate=finger_i_j2_rot_mat_wrt_origin,
                vector_in_prev_frame=pre_dip_vec,
                rotation_matrix_to_rearrange_coordinate=self._rot_mat_for_fingers_j3,
                angle_range_of_two_joints=self._angles_ranges[2:],
                axis_to_get_the_opposite_if_angle_exceed_the_limit_of_two_joints=[None, None],
                get_the_opposite_of_two_joints=[False, False],
                limit_angle_of_two_joints=[False, False],
                calculate_the_second_joint=False
            )

            finger_i_angle_j3 = j3_j4_results["older_brother_angle"]
            finger_i_j3_rot_mat_wrt_origin = j3_j4_results["older_brother_rot_mat_wrt_origin"]
            finger_i_j3_rot_mat_wrt_j2 = j3_j4_results["older_brother_rot_mat_wrt_parent"]
            finger_i_current_dip_vec = j3_j4_results["vector_in_current_frame"].copy()

            self._previous_landmark_vectors_dict[finger_name]["pre_dip_vec"] = finger_i_current_dip_vec

            # In robot, its finger joint 1 is our finger joint 2, and its finger joint 2
            # is our finger joint 1 (EXCEPT FOR THUMB FINGER).
            angles = (finger_i_angle_j2, finger_i_angle_j1, finger_i_angle_j3)
            coordinates_wrt_origin = (finger_i_j2_rot_mat_wrt_origin, 
                finger_i_j1_rot_mat_wrt_origin,
                finger_i_j3_rot_mat_wrt_origin)
            coordinates_wrt_parent = (finger_i_j2_rot_mat_wrt_j1, 
                finger_i_j1_rot_mat_wrt_last_coordinate_of_arm,
                finger_i_j3_rot_mat_wrt_j2)

            lefthand_angle_rs[finger_name] = {
                "angles": angles, 
                "coordinates_wrt_origin": coordinates_wrt_origin,
                "coordinates_wrt_parent": coordinates_wrt_parent
            }
        
        return lefthand_angle_rs 