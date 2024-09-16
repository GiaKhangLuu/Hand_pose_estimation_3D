"""
Currently support for left arm only
"""
import numpy as np
import traceback

elbow_vector_in_prev_frame = None
self_distance_threshold = 0.03

joint1_max = 86
joint1_min = -195

def set_joint1_rot_mat(x_unit_joint1):
    """
    TODO: doc.
    """
    y_unit_joint1 = np.array([-x_unit_joint1[1], x_unit_joint1[0], x_unit_joint1[2]])
    z_unit_joint1 = np.cross(x_unit_joint1, y_unit_joint1)

    assert 1 - np.linalg.norm(y_unit_joint1) < 1e-3
    assert 1 - np.linalg.norm(z_unit_joint1) < 1e-3

    rotation_matrix = np.hstack([x_unit_joint1[:, None], 
        y_unit_joint1[:, None], 
        z_unit_joint1[:, None]])

    return rotation_matrix

def calculate_joint1_angle(joint1_rot_mat):
    """
    TODO: doc.
    """
    joint1_angle = np.arctan2(joint1_rot_mat[0, 1], joint1_rot_mat[0, 0])
    joint1_angle = np.degrees(joint1_angle)

    assert -180 <= joint1_angle <= 180

    return joint1_angle

def map_to_rviz(joint1_angle):
    """
    TODO: doc.
    """
    if -90 <= joint1_angle <= 180:
        joint1_angle = round(-joint1_angle)
    else:
        joint1_angle = round(-joint1_angle - 360)
    return joint1_angle

def calculate_angle_j1(XYZ_landmarks, landmark_dictionary):
    """
    TODO: doc.
    """
    global elbow_vector_in_prev_frame
    global joint1_max
    global joint1_min
    global self_distance_threshold

    x_joint1 = XYZ_landmarks[landmark_dictionary.index("left elbow")].copy()
    x_joint1 *= (-1)
    x_joint1[-1] = 0
    x_joint1_length = np.linalg.norm(x_joint1)
    if x_joint1_length < self_distance_threshold:
        x_joint1 = elbow_vector_in_prev_frame.copy()
        x_joint1 *= (-1)
        x_joint1[-1] = 0
    else:
        elbow_vector_in_prev_frame = XYZ_landmarks[landmark_dictionary.index("left elbow")].copy()

    x_unit_joint1 = x_joint1 / np.linalg.norm(x_joint1)
    joint1_rot_mat = set_joint1_rot_mat(x_unit_joint1)
    joint1_angle = calculate_joint1_angle(joint1_rot_mat)
    joint1_angle = map_to_rviz(joint1_angle)

    if joint1_angle > joint1_max or joint1_angle < joint1_min: 
        x_unit_joint1[1] = -x_unit_joint1[1]
        joint1_rot_mat = set_joint1_rot_mat(x_unit_joint1)
        joint1_angle = calculate_joint1_angle(joint1_rot_mat)
        joint1_angle = map_to_rviz(joint1_angle)

    return joint1_angle, joint1_rot_mat 

def calculate_angle_j2(XYZ_landmarks, landmark_dictionary, joint1_rot_mat, joint1_angle): 
    """
    TODO: doc.
    """
    global elbow_vector_in_prev_frame
    global self_distance_threshold

    joint1_rot_mat_inv = joint1_rot_mat.T
    x_joint2 = XYZ_landmarks[landmark_dictionary.index("left elbow")].copy()
    x_joint2 *= (-1)
    #x_joint2[1] = 0
    x_joint2_length = np.linalg.norm(x_joint2)

    if (x_joint2_length < self_distance_threshold or joint1_angle == -90):
        x_joint2 = elbow_vector_in_prev_frame.copy()
        x_joint2 *= (-1)
        #x_joint2[1] = 0
    else:
        elbow_vector_in_prev_frame = XYZ_landmarks[landmark_dictionary.index("left elbow")].copy()

    x_unit_joint2 = x_joint2 / np.linalg.norm(x_joint2)
    x_unit_joint2_wrt_joint1 = np.matmul(joint1_rot_mat_inv, x_unit_joint2)
    x_unit_joint2_wrt_joint1[1] = 0
    x_unit_joint2_wrt_joint1 = x_unit_joint2_wrt_joint1 / np.linalg.norm(x_unit_joint2_wrt_joint1)
    z_unit_joint2_wrt_joint1 = np.array([-x_unit_joint2_wrt_joint1[-1], 
        x_unit_joint2_wrt_joint1[1], 
        x_unit_joint2_wrt_joint1[0]])
    y_unit_joint2_wrt_joint1 = np.cross(z_unit_joint2_wrt_joint1, x_unit_joint2_wrt_joint1)

    assert 1 - np.linalg.norm(y_unit_joint2_wrt_joint1) < 1e-3
    assert 1 - np.linalg.norm(z_unit_joint2_wrt_joint1) < 1e-3

    joint2_rot_mat_wrt_joint1 = np.hstack([x_unit_joint2_wrt_joint1[:, None], 
        y_unit_joint2_wrt_joint1[:, None], 
        z_unit_joint2_wrt_joint1[:, None]])

    joint2_angle = -np.arcsin(joint2_rot_mat_wrt_joint1[0, -1])
    joint2_angle = np.degrees(joint2_angle)

    joint2_angle = -joint2_angle

    joint2_rot_mat = np.matmul(joint1_rot_mat, joint2_rot_mat_wrt_joint1)

    return joint2_angle, joint2_rot_mat, joint2_rot_mat_wrt_joint1
