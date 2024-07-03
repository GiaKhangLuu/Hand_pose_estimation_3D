import numpy as np
from scipy.spatial.transform import Rotation as R

def calculate_angle_j2(XYZ_landmarks):
    a = np.array([1, 0, 0])
    b = XYZ_landmarks[1]
    b = b * [1, 0, 1]

    dot = np.sum(a * b)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = dot / (a_norm * b_norm)
    angle_j2 = np.degrees(np.arccos(cos))

    angle_j2 = 180 - angle_j2

    c = np.cross(b, a)
    ref = c[1] + 1e-9
    signs = ref / np.absolute(ref)
    angle_j2 *= signs

    return angle_j2


def get_angles_between_joints(XYZ_landmarks, landmark_dictionary):

    # Joint 1
    #a = np.array([1, 0, 0])
    #b = XYZ_landmarks[1]
    #b = b * [1, 1, 0]

    #dot = np.sum(a * b)
    #a_norm = np.linalg.norm(a)
    #b_norm = np.linalg.norm(b)
    #cos = dot / (a_norm * b_norm)
    #angle = np.degrees(np.arccos(cos))

    #angle = 180 - angle

    #c = np.cross(b, a)
    #ref = c[-1] + 1e-9
    #signs = ref / np.absolute(ref)

    #angle *= signs

    # -------------------------

    # Joint 2 
    angle_j2 = calculate_angle_j2(XYZ_landmarks)

    # ---------------------

    # Joint 3
    #rot_mat = R.from_euler('y', angle_j2, degrees=True).as_matrix()
    #z_new = np.matmul(rot_mat, [0, 0, 1])
    #b = XYZ_landmarks[landmark_dictionary.index("left wrist")] - XYZ_landmarks[landmark_dictionary.index("left elbow")]
    #ref_vec = np.cross(b, z_new)
    #a = np.cross(z_new, ref_vec)
    #a = np.cross(XYZ_landmarks[landmark_dictionary.index("left elbow")], z_new)

    rot_mat = R.from_euler('y', angle_j2, degrees=True).as_matrix()
    z_new = np.matmul(rot_mat, [0, 0, 1])
    y_new = XYZ_landmarks[landmark_dictionary.index("left elbow")]
    x_new = np.cross(y_new, z_new)

    x_new = x_new / np.linalg.norm(x_new)
    y_new = y_new / np.linalg.norm(y_new)
    z_new = z_new / np.linalg.norm(z_new)

    trans_mat = np.array([x_new, y_new, z_new])
    trans_mat_inv = np.linalg.inv(trans_mat)

    b = XYZ_landmarks[landmark_dictionary.index("left wrist")] - XYZ_landmarks[landmark_dictionary.index("left elbow")]
    b = np.matmul(trans_mat_inv, b)
    a = b * [1, 1, 0]
    
    dot = np.sum(a * b)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = dot / (a_norm * b_norm)
    angle = np.degrees(np.arccos(cos))

    #b_in_world = np.matmul(trans_mat, b)
    #a_in_world = np.matmul(trans_mat, a)
    c = np.cross(b, a)
    d = np.cross(a, c)
    print(d)
    #c = np.matmul(trans_mat, c)
    #print("c: ", c)
    ref = c[1] + 1e-9
    signs = ref / np.absolute(ref)

    angle *= signs

    print("angle: ", angle)
    # -------------------

    # Joint 4
    #a = np.array([0, 1, 0])
    #b = XYZ_landmarks[landmark_dictionary.index("left wrist")] - XYZ_landmarks[landmark_dictionary.index("left elbow")]
    #b = b * [1, 1, 0]
    #dot = np.sum(a * b)
    #a_norm = np.linalg.norm(a)
    #b_norm = np.linalg.norm(b)
    #cos = dot / (a_norm * b_norm)
    #angle = np.degrees(np.arccos(cos))

    #c = np.cross(b, a)
    #ref = c[-1] + 1e-9
    #signs = ref / np.absolute(ref)

    #angle *= signs

    # --------------------

    # Joint 5
    #a = np.array([0, 0, 1])
    #b = XYZ_landmarks[landmark_dictionary.index("left pinky")] - XYZ_landmarks[landmark_dictionary.index("left index")]
    #b = b * [1, 0, 1]
    #dot = np.sum(a * b)
    #a_norm = np.linalg.norm(a)
    #b_norm = np.linalg.norm(b)
    #cos = dot / (a_norm * b_norm)
    #angle = np.degrees(np.arccos(cos))

    
    return angle

