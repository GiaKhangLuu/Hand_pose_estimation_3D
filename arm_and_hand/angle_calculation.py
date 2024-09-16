"""
Currently support for left arm only
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

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

def calculate_elbow_coordinate_wrt_origin(XYZ_landmarks, landmark_dictionary, shoulder_coordinate_wrt_origin):
    """
    Input:
        XYZ_landmarks.shape = (M, N), M = (9 for arm) + (21 for hand)), N = 3 = #features
        shoulder_coordinate_wrt_origin = (N, O), N = 3 = #features, O = #vectors (x_w_sh, y_w_sh, z_w_sh)
    Return:
        elbow_coordinate_wrt_origin = (N, O), N = 3 = #features, O = #vectors (x_w_el, y_w_el, z_w_el)
    """
    y_elbow = XYZ_landmarks[landmark_dictionary.index("WRIST")] - XYZ_landmarks[landmark_dictionary.index("left elbow")]
    z_shoulder = shoulder_coordinate_wrt_origin[:, 2]

    try:
        x_elbow = np.cross(y_elbow, z_shoulder)
        z_elbow = np.cross(x_elbow, y_elbow)

        x_elbow_unit = x_elbow / np.linalg.norm(x_elbow)
        y_elbow_unit = y_elbow / np.linalg.norm(y_elbow)
        z_elbow_unit = z_elbow / np.linalg.norm(z_elbow)

        elbow_coordinate_wrt_origin = np.array([x_elbow_unit, y_elbow_unit, z_elbow_unit])  # (O, 3), O = number of vectors (xyz)
        elbow_coordinate_wrt_origin = np.transpose(elbow_coordinate_wrt_origin)  # (3, O), O = number of vectors (xyz)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    
    return elbow_coordinate_wrt_origin 

def calculate_elbow_coordinate_wrt_shoulder(shoulder_coordinate_wrt_origin, elbow_coordinate_wrt_origin):
    """
    As the equation, elbow_coordinate_wrt_shoulder = (shoulder_coordinate_wrt_origin)^(-1) @ elbow_coordinate_wrt_origin
    Input:
        shoulder_coordinate_wrt_origin = (N, O), N = 3 = #features, O = #vectors (x_w_sh, y_w_sh, z_w_sh)
        elbow_coordinate_wrt_origin = (N, O), N = 3 = #features, O = #vectors (x_w_el, y_w_el, z_w_el)
    Return:
        elbow_coordinate_wrt_shoulder = (N, O), N = 3 = #features, O = #vectors (x_sh_el, y_sh_el, z_sh_el)
    """
    try:
        shoulder_coordinate_wrt_origin_inv = np.linalg.inv(shoulder_coordinate_wrt_origin)
        elbow_coordinate_wrt_shoulder = np.matmul(shoulder_coordinate_wrt_origin_inv, elbow_coordinate_wrt_origin)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    return elbow_coordinate_wrt_shoulder 

def calculate_wrist_coordinate_wrt_origin(XYZ_landmarks, landmark_dictionary, elbow_coordinate_wrt_origin):
    """
    Input:
        XYZ_landmarks.shape = (M, N), M = (9 for arm) + (21 for hand)), N = 3 = #features
        elbow_coordinate_wrt_origin = (N, O), N = 3 = #features, O = #vectors (x_w_el, y_w_el, z_w_el)
    Return:
        wrist_coordinate_wrt_origin = (N, O), N = 3 = #features, O = #vectors (x_w_wr, y_w_wr, z_w_wr)
    """
    u_wrist = XYZ_landmarks[landmark_dictionary.index("INDEX_FINGER_MCP"), :] - XYZ_landmarks[landmark_dictionary.index("WRIST"), :]
    v_wrist = XYZ_landmarks[landmark_dictionary.index("MIDDLE_FINGER_MCP"), :] - XYZ_landmarks[landmark_dictionary.index("WRIST"), :]
    y_elbow = elbow_coordinate_wrt_origin[:, 1]

    try:
        x_wrist = np.cross(v_wrist, u_wrist)
        z_wrist = np.cross(x_wrist, y_elbow)
        y_wrist = np.cross(z_wrist, x_wrist)

        x_wrist_unit = x_wrist / np.linalg.norm(x_wrist)
        y_wrist_unit = y_wrist / np.linalg.norm(y_wrist)
        z_wrist_unit = z_wrist / np.linalg.norm(z_wrist)

        wrist_coordinate_wrt_origin = np.array([x_wrist_unit, y_wrist_unit, z_wrist_unit])  # (O, 3), O = number of vectors (xyz)
        wrist_coordinate_wrt_origin = np.transpose(wrist_coordinate_wrt_origin)  # (3, O), O = number of vectors (xyz)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    
    return wrist_coordinate_wrt_origin 

def calculate_wrist_coordinate_wrt_elbow(elbow_coordinate_wrt_origin, wrist_coordinate_wrt_origin):
    """
    As the equation, wrist_coordinate_wrt_elbow = (elbow_coordinate_wrt_origin)^(-1) @ wrist_coordinate_wrt_origin
    Input:
        elbow_coordinate_wrt_origin = (N, O), N = 3 = #features, O = #vectors (x_w_el, y_w_el, z_w_el)
        wrist_coordinate_wrt_origin = (N, O), N = 3 = #features, O = #vectors (x_w_wr, y_w_wr, z_w_wr)
    Return:
        wrist_coordinate_wrt_elbow = (N, O), N = 3 = #features, O = #vectors (x_el_wr, y_el_wr, z_el_wr)
    """
    try:
        elbow_coordinate_wrt_origin_inv = np.linalg.inv(elbow_coordinate_wrt_origin)
        wrist_coordinate_wrt_elbow = np.matmul(elbow_coordinate_wrt_origin_inv, wrist_coordinate_wrt_origin)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    return wrist_coordinate_wrt_elbow 

def calculate_angle_j1(shoulder_coords_in_world_rot_mat):
    # Joint 1 rotates around z-axis
    a, b, angle_j1 = shoulder_coords_in_world_rot_mat.as_euler("xyz", degrees=True)
    print("{}-------{}-------{}".format(round(a, 2), round(b, 2), round(angle_j1, 2)))
    print("-----------")
    return angle_j1

def calculate_angle_j2(shoulder_coords_in_world_rot_mat):
    # Joint 2 rotates around y-axis
    _, angle_j2, _ = shoulder_coords_in_world_rot_mat.as_euler("xyz", degrees=True)
    return angle_j2

def calculate_angle_j3(elbow_coordinate_wrt_shoulder_rot_mat):
    # Joint 3 rotates around x-axis
    angle_j3, _, _ = elbow_coordinate_wrt_shoulder_rot_mat.as_euler("xyz", degrees=True)
    angle_j3 = angle_j3 * (-1)  # In order to be compatible with the robot in RVIZ, we need to HARDCODE to -angle_j3
    return angle_j3

def calculate_angle_j4(elbow_coordinate_wrt_shoulder_rot_mat):
    # Joint 4 rotates around z-axis
    _, _, angle_j4 = elbow_coordinate_wrt_shoulder_rot_mat.as_euler("xyz", degrees=True)
    angle_j4 = angle_j4 * (-1)  # In order to be compatible with the robot in RVIZ, we need to HARDCODE to -angle_j4
    return angle_j4

def calculate_angle_j5(wrist_coordinate_wrt_elbow_rot_mat,  wrist_coordinate_wrt_elbow):
    """
    According to the scipy documentation, the pitch value ranges between -90 and 90.
    When the angle of joint 5 exceeds 90 degree in real world, the value we got from
    as_euler() is lower than 90 degree. Therefore, we have to get the direction of
    x-axis and offset to the angle of joint 5.
    """

    # Joint 5 rotates around y-axis
    _, angle_j5, _ = wrist_coordinate_wrt_elbow_rot_mat.as_euler("xyz", degrees=True)

    z_el_wr = wrist_coordinate_wrt_elbow[:, 2]
    sign = np.sign(z_el_wr[2])
    if sign == 1:
        angle_j5 = angle_j5
    elif sign == 0:
        angle_j5 = 90
    else:
        angle_j5 = 180 - angle_j5

    return angle_j5

def calculate_angle_j6(wrist_coordinate_wrt_elbow_rot_mat):
    # Joint 6 rotates around z-axis
    _, _, angle_j6 = wrist_coordinate_wrt_elbow_rot_mat.as_euler("xyz", degrees=True)
    return angle_j6

def get_angles_between_joints(XYZ_landmarks, landmark_dictionary, original_xyz):
    """
    Input:
        XYZ_landmarks.shape = (M, N), M = (9 for arm) + (21 for hand)), N = 3 = #features
        original_xyz = (N, O), N = 3 = #features, O = #vectors (xyz)
    """

    # Joint 1 and Joint 2
    shoulder_coords_in_world = calculate_rotation_matrix_to_compute_angle_of_j1_and_j2(XYZ_landmarks,  
        landmark_dictionary, original_xyz)  # (3, O), O = number of vectors (xyz)
    shoulder_coords_in_world_rot_mat = R.from_matrix(shoulder_coords_in_world)
    angle_j1 = calculate_angle_j1(shoulder_coords_in_world_rot_mat)
    angle_j2 = calculate_angle_j2(shoulder_coords_in_world_rot_mat)

    # Joint 3 and Joint 4
    elbow_coordinate_wrt_origin = calculate_elbow_coordinate_wrt_origin(XYZ_landmarks,
        landmark_dictionary, shoulder_coords_in_world)  # (3, O), O = number of vectors (xyz)
    elbow_coordinate_wrt_shoulder = calculate_elbow_coordinate_wrt_shoulder(shoulder_coords_in_world,
        elbow_coordinate_wrt_origin)  # (3, O), O = number of vectors (xyz)
    elbow_coordinate_wrt_shoulder_rot_mat = R.from_matrix(elbow_coordinate_wrt_shoulder)
    angle_j3 = calculate_angle_j3(elbow_coordinate_wrt_shoulder_rot_mat)
    angle_j4 = calculate_angle_j4(elbow_coordinate_wrt_shoulder_rot_mat)
    
    # Joint 5 and Joint 6
    wrist_coordinate_wrt_origin = calculate_wrist_coordinate_wrt_origin(XYZ_landmarks,
        landmark_dictionary, elbow_coordinate_wrt_origin)  # (3, O), O = number of vectors (xyz)
    wrist_coordinate_wrt_elbow = calculate_wrist_coordinate_wrt_elbow(elbow_coordinate_wrt_origin,
        wrist_coordinate_wrt_origin)  # (3, O), O = number of vectors (xyz)
    wrist_coordinate_wrt_elbow_rot_mat = R.from_matrix(wrist_coordinate_wrt_elbow)
    angle_j5 = calculate_angle_j5(wrist_coordinate_wrt_elbow_rot_mat, wrist_coordinate_wrt_elbow)
    angle_j6 = calculate_angle_j6(wrist_coordinate_wrt_elbow_rot_mat)

    return angle_j1, angle_j2, angle_j3, angle_j4, angle_j5, angle_j6

