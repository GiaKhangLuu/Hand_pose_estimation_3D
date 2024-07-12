"""
Currently support for left arm only
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

def calculate_angle_j1(XYZ_landmarks, landmark_dictionary):
    a = np.array([1, 0, 0])  # ref. vector is vector x_unit (1, 0, 0)
    b = XYZ_landmarks[landmark_dictionary.index("left elbow")]
    b = b * [1, 1, 0]  # vector b projects to Oxy plane (rotate around z) => z = 0

    dot = np.sum(a * b)
    unit_a = np.linalg.norm(a)
    unit_b = np.linalg.norm(b)
    cos = dot / (unit_a * unit_b)
    angle_j1 = np.degrees(np.arccos(cos))

    angle_j1 = 180 - angle_j1

    c = np.cross(b, a)
    ref = c[-1] + 1e-9  # get the sign of element z of vector c
    signs = ref / np.absolute(ref)

    angle_j1 *= signs

    return a, b, angle_j1

def calculate_angle_j2(XYZ_landmarks, landmark_dictionary):
    a = np.array([1, 0, 0])  # ref. vector is vector x_unit (1, 0, 0) 
    b = XYZ_landmarks[landmark_dictionary.index("left elbow")]
    b = b * [1, 0, 1]  # vector b projects to Oxz plane Oxz plane (rotate around y) => y = 0

    dot = np.sum(a * b)
    unit_a = np.linalg.norm(a)
    unit_b = np.linalg.norm(b)
    cos = dot / (unit_a * unit_b)
    angle_j2 = np.degrees(np.arccos(cos))

    angle_j2 = 180 - angle_j2

    c = np.cross(b, a)
    ref = c[1] + 1e-9  # get the sign of element y of vector c
    signs = ref / np.absolute(ref)
    angle_j2 *= signs

    return a, b, angle_j2

def calculate_rotation_matrix_to_compute_angle_of_j3_and_j4(XYZ_landmarks, angle_j1, angle_j2, landmark_dictionary):
    """
    In order to calculate angle_j3, we have to put angle_j1 and angle_j2. Because
    when joint_1 and joint_2 rotate, we have new coordinate.
    """

    rot_mat = R.from_euler("yz", [angle_j2, angle_j1], degrees=True).as_matrix() 

    z_new = np.matmul(rot_mat, [0, 0, 1])
    y_new = XYZ_landmarks[landmark_dictionary.index("left elbow")]
    x_new = np.cross(y_new, z_new)

    x_unit = x_new / np.linalg.norm(x_new)
    y_unit = y_new / np.linalg.norm(y_new)
    z_unit = z_new / np.linalg.norm(z_new)

    trans_mat = np.array([x_unit, y_unit, z_unit]) 
    trans_mat = np.transpose(trans_mat)
    trans_mat_inv = np.linalg.inv(trans_mat)

    return trans_mat, trans_mat_inv

def calculate_angle_j3(XYZ_landmarks, trans_mat_inv, landmark_dictionary):
    """
    The values (x, y, z) of vector `a` and `b` in this function is belong to 
    the new coordinate (when joint1 and joint2) rotate, not the original 
    coordinate (shoulder).
    Input:
        trans_mat_inv: Get the value of joint in this coordinate to compute angle
    """
    b = XYZ_landmarks[landmark_dictionary.index("WRIST")] - XYZ_landmarks[landmark_dictionary.index("left elbow")]
    b = np.matmul(trans_mat_inv, b.T)  # get b in the new coordinate
    b = b.T  
    a = b * [1, 1, 0]  # vector a is the projection of vector b to Oxy plane => z = 0
    
    dot = np.sum(a * b)
    unit_a = np.linalg.norm(a)
    unit_b = np.linalg.norm(b)
    cos = dot / (unit_a * unit_b)
    angle_j3 = np.degrees(np.arccos(cos))

    c = np.cross(a, b)
    ref = c[1] + 1e-9  # get the sign of element y of vector c
    signs = ref / np.absolute(ref)

    angle_j3 *= signs

    return a, b, angle_j3

def calculate_angle_j4(XYZ_landmarks, trans_mat, trans_mat_inv, landmark_dictionary):
    """
    The values (x, y, z) of vector `a` and `b` in this function is belong to 
    the new coordinate (when joint1 and joint2) rotate, not the original 
    coordinate (shoulder).
    Input:
        trans_mat_inv: Get the value of joint in this coordinate to compute angle
    """
    b = XYZ_landmarks[landmark_dictionary.index("WRIST")] - XYZ_landmarks[landmark_dictionary.index("left elbow")]
    b = np.matmul(trans_mat_inv, b.T)  # get b in the new coordinate
    b = b.T  
    b = b * [1, 1, 0]  # vector b projects to Oxy plane => z = 0

    a = trans_mat[:, 0]  # ref. vector is vector x_unit (1, 0, 0) (IN NEW COORDINATE) 
    a = np.matmul(trans_mat_inv, a.T)
    a = a.T
    
    dot = np.sum(a * b)
    unit_a = np.linalg.norm(a)
    unit_b = np.linalg.norm(b)
    cos = dot / (unit_a * unit_b)
    angle_j4 = np.degrees(np.arccos(cos))

    c = np.cross(b, a)
    ref = c[-1] + 1e-9  # get the sign of element z of vector c
    signs = ref / np.absolute(ref)
    angle_j4 *= signs

    return a, b, angle_j4

def get_angles_between_joints(XYZ_landmarks, landmark_dictionary):

    # Joint 1
    _, _, angle_j1 = calculate_angle_j1(XYZ_landmarks, landmark_dictionary)

    # Joint 2 
    _, _, angle_j2 = calculate_angle_j2(XYZ_landmarks, landmark_dictionary)

    trans_mat, trans_mat_inv = calculate_rotation_matrix_to_compute_angle_of_j3_and_j4(XYZ_landmarks, angle_j1, angle_j2, landmark_dictionary)

    # Joint 3
    _, _, angle_j3 = calculate_angle_j3(XYZ_landmarks, trans_mat_inv, landmark_dictionary)

    # Joint 4
    _, _, angle_j4 = calculate_angle_j4(XYZ_landmarks, trans_mat, trans_mat_inv, landmark_dictionary)

    # Joint 5
    #a = np.array([0, 0, 1])
    #b = XYZ_landmarks[landmark_dictionary.index("left pinky")] - XYZ_landmarks[landmark_dictionary.index("left index")]
    #b = b * [1, 0, 1]
    #dot = np.sum(a * b)
    #a_norm = np.linalg.norm(a)
    #b_norm = np.linalg.norm(b)
    #cos = dot / (a_norm * b_norm)
    #angle = np.degrees(np.arccos(cos))

    return angle_j4

