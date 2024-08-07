"""
Currently support for left arm only
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

#def calculate_angle_j1(XYZ_landmarks, landmark_dictionary):
    #a = np.array([1, 0, 0])  # ref. vector is vector x_unit (1, 0, 0)
    #b = XYZ_landmarks[landmark_dictionary.index("left elbow")]
    #b = b * [1, 1, 0]  # vector b projects to Oxy plane (rotate around z) => z = 0

    #dot = np.sum(a * b)
    #unit_a = np.linalg.norm(a)
    #unit_b = np.linalg.norm(b)
    #cos = dot / (unit_a * unit_b)
    #angle_j1 = np.degrees(np.arccos(cos))

    #angle_j1 = 180 - angle_j1

    #c = np.cross(b, a)
    #ref = c[-1] + 1e-9  # get the sign of element z of vector c
    #signs = ref / np.absolute(ref)

    #angle_j1 *= signs

    #return a, b, angle_j1

#def calculate_angle_j2(XYZ_landmarks, landmark_dictionary):
    #a = np.array([1, 0, 0])  # ref. vector is vector x_unit (1, 0, 0) 
    #b = XYZ_landmarks[landmark_dictionary.index("left elbow")]
    #b = b * [1, 0, 1]  # vector b projects to Oxz plane Oxz plane (rotate around y) => y = 0

    #dot = np.sum(a * b)
    #unit_a = np.linalg.norm(a)
    #unit_b = np.linalg.norm(b)
    #cos = dot / (unit_a * unit_b)
    #angle_j2 = np.degrees(np.arccos(cos))

    #angle_j2 = 180 - angle_j2

    #c = np.cross(b, a)
    #ref = c[1] + 1e-9  # get the sign of element y of vector c
    #signs = ref / np.absolute(ref)
    #angle_j2 *= signs

    #return a, b, angle_j2

def calculate_rotation_matrix_to_compute_angle_of_j1_and_j2(XYZ_landmarks, landmark_dictionary, XYZ_origin):
    """
    Input:
        XYZ_landmarks.shape = (M, N), M = (9 for arm) + (21 for hand)), N = 3 = #features
        XYZ_origin = (N, O), N = 3 = #features, O = #vectors (xyz)
    Return:
        shoulder_coords_in_world = (N, O), N = 3 = #features, O = #vectors (xyz)
    """
    x_shoulder =  XYZ_landmarks[landmark_dictionary.index("left elbow")] * -1
    y_unit, z_unit = XYZ_origin[:, 1], XYZ_origin[:, 2]

    try:
        y_shoulder = np.cross(z_unit, x_shoulder)
        z_shoulder = np.cross(x_shoulder, y_shoulder)

        x_shoulder_unit = x_shoulder / np.linalg.norm(x_shoulder)
        y_shoulder_unit = y_shoulder / np.linalg.norm(y_shoulder)
        z_shoulder_unit = z_shoulder / np.linalg.norm(z_shoulder)

        shoulder_coords_in_world = np.array([x_shoulder_unit, y_shoulder_unit, z_shoulder_unit])  # (O, 3), O = number of vectors (xyz)
        shoulder_coords_in_world = np.transpose(shoulder_coords_in_world)  # (3, O), O = number of vectors (xyz)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    return shoulder_coords_in_world

def calculate_angle_j1(shoulder_coords_in_world_rot_mat):
    # Joint 1 rotates around z-axis
    _, _, angle_j1 = shoulder_coords_in_world_rot_mat.as_euler("xyz", degrees=True)
    return angle_j1

def calculate_angle_j2(shoulder_coords_in_world_rot_mat):
    # Joint 2 rotates around y-axis
    _, angle_j2, _ = shoulder_coords_in_world_rot_mat.as_euler("xyz", degrees=True)
    return angle_j2

def calculate_rotation_matrix_to_compute_angle_of_j3_and_j4(XYZ_landmarks, angle_j1, angle_j2, landmark_dictionary):
    """
    In order to calculate angle_j3, we have to put angle_j1 and angle_j2. Because
    when joint_1 and joint_2 rotate, we have new coordinate.
    """

    #rot_mat = R.from_euler("yz", [angle_j2, angle_j1], degrees=True).as_matrix() 
    rot_mat = R.from_euler("zy", [angle_j1, angle_j2], degrees=True).as_matrix() 

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

def calculate_angle_j3(XYZ_landmarks, trans_mat, trans_mat_inv, landmark_dictionary):
    """
    The values (x, y, z) of vector `a` and `b` in this function is belong to 
    the new coordinate (when joint1 and joint2) rotate, not the original 
    coordinate (shoulder).
    Input:
        trans_mat_inv: Get the value of joint in this coordinate to compute angle
    """
    b = XYZ_landmarks[landmark_dictionary.index("WRIST")] - XYZ_landmarks[landmark_dictionary.index("left elbow")]
    b = np.matmul(trans_mat_inv, b)  # get b in the new coordinate
    b = b * [1, 0, 1]  

    a = trans_mat[:, 0]  # ref. vector is vector x_unit (1, 0, 0) (IN NEW COORDINATE) 
    a = np.matmul(trans_mat_inv, a)
    
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
    b = np.matmul(trans_mat_inv, b)  # get b in the new coordinate
    b = b * [1, 1, 0]  # vector b projects to Oxy plane => z = 0

    a = trans_mat[:, 0]  # ref. vector is vector x_unit (1, 0, 0) (IN NEW COORDINATE) 
    a = np.matmul(trans_mat_inv, a)
    
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

def calculate_elbow_coords(XYZ_landmarks, landmark_dictionary, shoulder_coords_in_world, angle_j3, angle_j4):
    """
    Input:
        shoulder_coords: Shoudler coordinate
            shape = (3, 3)
            x = shoulder_coords[:, 0], y = shoulder_coords[:, 1], z = shoulder_coords[:, 2]
    """

    angle_j4 = -angle_j4  # In order to be compatible with the robot in RVIZ, we need to HARDCODE to -angle_j4
    elbow_coords_in_shoulder = R.from_euler("zy", [angle_j4, angle_j3], degrees=True).as_matrix()
    #elbow_coords_in_shoulder = R.from_euler("yz", [angle_j3, angle_j4], degrees=True).as_matrix()
    elbow_coords_in_world = np.matmul(shoulder_coords_in_world, elbow_coords_in_shoulder)    

    return elbow_coords_in_world, elbow_coords_in_shoulder

def calculate_wrist_coords(XYZ_landmarks, landmark_dictionary):
    # We must ensure that x, y and z from wrist_coords have the same direction as x, y and z from elbow_coords
    u_wrist = XYZ_landmarks[landmark_dictionary.index("INDEX_FINGER_MCP"), :] - XYZ_landmarks[landmark_dictionary.index("WRIST"), :]
    x_wrist = XYZ_landmarks[landmark_dictionary.index("MIDDLE_FINGER_MCP"), :] - XYZ_landmarks[landmark_dictionary.index("WRIST"), :]

    y_wrist = np.cross(u_wrist, x_wrist)
    z_wrist = np.cross(x_wrist, y_wrist)

    x_wrist_unit = x_wrist / np.linalg.norm(x_wrist)
    y_wrist_unit = y_wrist / np.linalg.norm(y_wrist)
    z_wrist_unit = z_wrist / np.linalg.norm(z_wrist)

    wrist_coords_in_world = np.array([x_wrist_unit, y_wrist_unit, z_wrist_unit])  # (N, 3), N = number of vectors
    wrist_coords_in_world = np.transpose(wrist_coords_in_world)  # (3, N), N = number of vectors

    return wrist_coords_in_world

def calculate_angle_j5(wrist_coords_in_elbow_rot_mat):
    # Joint 5 rotates around x-axis 
    angle_j5, _, _ = wrist_coords_in_elbow_rot_mat.as_euler("xyz", degrees=True)
    return angle_j5

def calculate_angle_j6(wrist_coords_in_elbow_rot_mat):
    # Joint 6 rotates around z-axis
    _, _, angle_j6 = wrist_coords_in_elbow_rot_mat.as_euler("xyz", degrees=True)
    return angle_j6

def calculate_wrist_coords_in_elbow(wrist_coords_in_world, elbow_coords_in_world):
    """
    As the equation, wrist_coords_in_elbow = (elbow_coords_in_world)^(-1) @ wrist_coords_in_world
    """
    elbow_coords_in_world_inv = np.linalg.inv(elbow_coords_in_world)
    wrist_coords_in_elbow = np.matmul(elbow_coords_in_world_inv, wrist_coords_in_world)
    return wrist_coords_in_elbow

def get_angles_between_joints(XYZ_landmarks, landmark_dictionary, original_xyz):
    """
    Input:
        XYZ_landmarks.shape = (M, N), M = (9 for arm) + (21 for hand)), N = 3 = #features
        original_xyz = (N, O), N = 3 = #features, O = #vectors (xyz)
    """

    # Joint 1 and Joint 2
    #_, _, angle_j1 = calculate_angle_j1(XYZ_landmarks, landmark_dictionary)
    shoulder_coords_in_world = calculate_rotation_matrix_to_compute_angle_of_j1_and_j2(XYZ_landmarks,  
        landmark_dictionary, original_xyz)  # (3, O), O = number of vectors (xyz)
    shoulder_coords_in_world_rot_mat = R.from_matrix(shoulder_coords_in_world)
    angle_j1 = calculate_angle_j1(shoulder_coords_in_world_rot_mat)
    angle_j2 = calculate_angle_j2(shoulder_coords_in_world_rot_mat)

    #_, _, angle_j2 = calculate_angle_j2(XYZ_landmarks, landmark_dictionary)

    # Joint 3 and Joint 4
    shoulder_rot_mat_in_w, shoulder_rot_mat_in_w_inv = calculate_rotation_matrix_to_compute_angle_of_j3_and_j4(XYZ_landmarks, angle_j1, angle_j2, landmark_dictionary)
    _, _, angle_j3 = calculate_angle_j3(XYZ_landmarks, shoulder_rot_mat_in_w, shoulder_rot_mat_in_w_inv, landmark_dictionary)
    _, _, angle_j4 = calculate_angle_j4(XYZ_landmarks, shoulder_rot_mat_in_w, shoulder_rot_mat_in_w_inv, landmark_dictionary)

    # Joint 5 and Joint 6
    """
    In order to calculate angles of joint_5 and joint_6, we follow these steps:
    1. Find the elbow coordinate. Got the elbow coordinate from the shoulder coordinate,    
        angles of joint_3 and joint_4.
    2. Find the wrist coordinate. We must ensure that (x_unit, y_unit, z_unit) 
        of wrist coordinate has the same direction as (x_unit, y_unit, z_unit)
        of elbow coordinate.
    3. Calculate the rotation matrix from elbow coordinate to wrist coordinate. 
        -> As the equation, wrist_coords_in_elbow = (elbow_coords_in_world)^(-1) @ wrist_coords_in_world
    4. Extract roll, pitch and yaw from that rotation matrix.
    """
    elbow_coords_in_world, _ = calculate_elbow_coords(XYZ_landmarks, landmark_dictionary, 
        shoulder_rot_mat_in_w, angle_j3, angle_j4)
    wrist_coords_in_world = calculate_wrist_coords(XYZ_landmarks, landmark_dictionary)  # (3, N), N = number of vectors
    wrist_coords_in_elbow = calculate_wrist_coords_in_elbow(wrist_coords_in_world, elbow_coords_in_world)
    rot_mat = R.from_matrix(wrist_coords_in_elbow)
    angle_j5 = calculate_angle_j5(rot_mat)
    angle_j6 = calculate_angle_j6(rot_mat)

    return angle_j5

