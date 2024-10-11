import numpy as np
from scipy.spatial.transform import Rotation as R
from left_finger_angle_calculator import LeftFingerAngleCalculator
from left_thumb_finger_angle_calculator import LeftThumbFingerAngleCalculator

# These five quaternion vectors get from ROS2, which are formatted: (x, y, z, w)
wrist_to_home_position_thumb_finger_quat = [-0.794, 0.078, 0.105, 0.593]
wrist_to_home_position_index_finger_quat = [0.595, 0.517, 0.465, 0.404]
wrist_to_home_position_middle_finger_quat = [0.534, 0.508, 0.489, 0.466]
wrist_to_home_position_ring_finger_quat = [0.466, 0.489, 0.508, 0.534]
wrist_to_home_position_pinky_finger_quat = [0.404, 0.465, 0.517, 0.595]

wrist_to_home_fingers_quat = [
    wrist_to_home_position_thumb_finger_quat, 
    wrist_to_home_position_index_finger_quat,
    wrist_to_home_position_middle_finger_quat,
    wrist_to_home_position_ring_finger_quat,
    wrist_to_home_position_pinky_finger_quat
]

last_coord_of_real_person_to_last_coord_in_rviz_rot_mat = R.from_euler("xz", [90, 180], degrees=True).as_matrix()

class LeftHandAngleCalculator():
    def __init__(self, num_chain, landmark_dictionary):
        # We keep capital words in order for the compatibility with landmark_dictionary
        self.fingers_name = ("THUMB", "INDEX", "MIDDLE", "RING", "PINKY")
        self.fingers_calculator = []
        for i, finger_name in enumerate(self.fingers_name):
            if finger_name == "THUMB":
                calculator = LeftThumbFingerAngleCalculator
            else:
                calculator = LeftFingerAngleCalculator
            left_finger_angle_calculator = calculator(
                num_chain=num_chain,
                finger_name=finger_name,
                landmark_dictionary=landmark_dictionary,
                last_coord_of_robot_to_home_position_of_finger_quat=wrist_to_home_fingers_quat[i],
                last_coord_of_real_person_to_last_coord_of_robot_in_rviz_rot_mat=last_coord_of_real_person_to_last_coord_in_rviz_rot_mat 
            )
            self.fingers_calculator.append(left_finger_angle_calculator)

    def __call__(self, XYZ_landmarks, parent_coordinate):
        merged_result_dict = dict()
        for finger_calculator, finger_name in zip(self.fingers_calculator, self.fingers_name):
            finger_result = finger_calculator(XYZ_landmarks, parent_coordinate.copy())
            merged_result_dict.update(finger_result)
        return merged_result_dict
