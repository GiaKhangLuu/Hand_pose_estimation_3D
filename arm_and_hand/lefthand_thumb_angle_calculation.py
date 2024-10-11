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

# This quaternion vector got from ROS2, which is formatted as (x, y, z, w)
wrist_to_home_position_thumb_finger_quat = [-0.794, 0.078, 0.105, 0.593]
j6_of_real_people_to_j6_in_rviz_rot_mat = R.from_euler("xz", [90, 180], degrees=True).as_matrix()
rot_mat_to_rearrange_finger_coord = np.eye(3)
rot_mat_for_fingers_j3 = np.eye(3)

bound = 2
thumb_joint1_max = 45 - bound
thumb_joint1_min = -74 + bound
thumb_joint2_max = 0 - bound
thumb_joint2_min = -90 + bound
thumb_joint3_max = 0 - bound
thumb_joint3_min = -90 + bound

class ThumbFingerAngles():
    def __init__(self, landmark_dictionary):
        """
        TODO: Doc.
        """

        self._rot_mat_to_cvt_last_coord_of_real_peoples_arm_to_last_coordinate_of_arm_in_rviz = j6_of_real_people_to_j6_in_rviz_rot_mat 
        self._rot_mat_to_rearrange_finger_coord = rot_mat_to_rearrange_finger_coord
        self._landmark_dictionary = landmark_dictionary
        self._rot_mat_to_align_last_coord_of_arm_to_finger_coords = R.from_quat(wrist_to_home_position_thumb_finger_quat).as_matrix()
        self._mapping_real_person_angle_to_robot_functions = (self._mapping_first_joint_angle_of_thumb, 
            self._mapping_second_joint_angle_of_thumb)
        self._thumb_angles_ranges = [[thumb_joint1_min, thumb_joint1_max],
            [thumb_joint2_min, thumb_joint2_max],
            [thumb_joint3_min, thumb_joint3_max],
            [None, None]]
        self._rot_mat_for_fingers_j3 = rot_mat_for_fingers_j3
        #self._previous_landmark_vector_dict = {
            #"pre_"
        #}

    def _mapping_first_joint_angle_of_thumb(self, angle):
        return angle

    def _mapping_second_joint_angle_of_thumb(self, angle):
        return angle