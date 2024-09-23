"""
Currently support for left arm only
"""
import numpy as np
import traceback
from scipy.spatial.transform import Rotation as R

shoulder_vector_in_prev_frame = None
elbow_vector_in_prev_frame = None
wrist_vector_in_prev_frame = None

rotation_matrix_for_shoulder = np.eye(3)
rotation_matrix_for_elbow = R.from_euler("xz", [90, -90], degrees=True).as_matrix()
rotation_matrix_for_wrist = R.from_euler("y", -90, degrees=True).as_matrix()

length_threshold = 0.03
axis_names = ['x', 'y', 'z']

joint1_max = 86
joint1_min = -195

joint2_max = 92
joint2_min = -3

joint3_max = 143
joint3_min = -143

joint4_max = 22
joint4_min = -91

joint5_max = 206
joint5_min = -115

joint6_max = 52
joint6_min = -69

def get_angle_and_rot_mats(XYZ_landmarks, 
    landmark_name,
    landmark_dictionary,
    parent_rot_mat,
    vector_in_prev_frame,
    calculate_angle_func,
    joint_name,
    axis_to_remove,
    rotation_matrix=None,
    axis_to_get_the_opposite_if_angle_exceed_the_limit=None,
    angle_range=None,
    get_the_opposite=False,
    limit_angle=False):
    """
    TODO: Doc.
    """
    global length_threshold
    global axis_names
    vector_in_current_frame_is_near_zero_after_project = False
    if rotation_matrix is not None:
        parent_rot_mat = parent_rot_mat @ rotation_matrix
    parent_rot_mat_inv = parent_rot_mat.T
    vec_landmark = get_landmark_vector(XYZ_landmarks, landmark_dictionary, landmark_name)
    if get_the_opposite:
        vec_landmark *= -1
    vec_landmark_wrt_parent = np.matmul(parent_rot_mat_inv, vec_landmark)
    vec_landmark_wrt_parent[axis_names.index(axis_to_remove)] = 0
    vec_landmark_wrt_parent_length = np.linalg.norm(vec_landmark_wrt_parent)

    if vec_landmark_wrt_parent_length < length_threshold:
        vec_landmark = vector_in_prev_frame.copy()
        if get_the_opposite:
            vec_landmark *= -1
        vec_landmark_wrt_parent = np.matmul(parent_rot_mat_inv, vec_landmark)
        vec_landmark_wrt_parent[axis_names.index(axis_to_remove)] = 0
        vec_landmark_wrt_parent_length = np.linalg.norm(vec_landmark_wrt_parent)
        vector_in_current_frame_is_near_zero_after_project = True

    unit_vec_landmark_wrt_parent = vec_landmark_wrt_parent / vec_landmark_wrt_parent_length 
    rot_mat_wrt_parent = create_rot_mat(unit_vec_landmark_wrt_parent, joint_name)
    angle = calculate_angle_func(rot_mat_wrt_parent)
    angle = cvt_angle_to_TomOSPC_angle(angle, joint_name)

    if limit_angle:
        assert angle_range is not None
        assert axis_to_get_the_opposite_if_angle_exceed_the_limit is not None
        if angle > angle_range[1] or angle < angle_range[0]:
            axis_idx = axis_names.index(axis_to_get_the_opposite_if_angle_exceed_the_limit)
            unit_vec_landmark_wrt_parent[axis_idx] = -unit_vec_landmark_wrt_parent[axis_idx]
        rot_mat_wrt_parent = create_rot_mat(unit_vec_landmark_wrt_parent, joint_name)
        angle = calculate_angle_func(rot_mat_wrt_parent)
        angle = cvt_angle_to_TomOSPC_angle(angle, joint_name)

    rot_mat_wrt_origin = np.matmul(parent_rot_mat, rot_mat_wrt_parent)

    result = {
        "angle": angle,
        "rot_mat_wrt_origin": rot_mat_wrt_origin,
        "rot_mat_wrt_parent": rot_mat_wrt_parent,
        "is_near_zero_after_project": vector_in_current_frame_is_near_zero_after_project
    }

    return result

def cvt_angle_to_TomOSPC_angle(angle, joint_name):
    """
    TODO: doc.
    Input:
        angle (float): angle computed by rotation matrix
        joint_name (Tuple): ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
    Ouptut:
        tomospc_angle (int): shape = (3, 3); each column is a vector (x, y, z) (horizontally stack)
    """

    assert joint_name in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")

    if joint_name == "joint1":
        tomospc_angle = joint1_angle_to_TomOSPC_angle(angle)
    elif joint_name == "joint2":
        tomospc_angle = joint2_angle_to_TomOSPC_angle(angle)
    elif joint_name == "joint3":
        tomospc_angle = joint3_angle_to_TomOSPC_angle(angle)
    elif joint_name == "joint4":
        tomospc_angle = joint4_angle_to_TomOSPC_angle(angle)
    elif joint_name == "joint5":
        tomospc_angle = joint5_angle_to_TomOSPC_angle(angle)
    else:
        tomospc_angle = joint6_angle_to_TomOSPC_angle(angle)
    
    return tomospc_angle

def joint1_angle_to_TomOSPC_angle(joint1_angle):
    """
    TODO: doc. 
    Input: 
        angle (float):
    Output:
        tomospc_angle_j1 (int): 
    """
    if -90 <= joint1_angle <= 180:
        tomospc_angle_j1 = round(-joint1_angle)
    else:
        tomospc_angle_j1 = round(-joint1_angle - 360)
    return tomospc_angle_j1 

def joint2_angle_to_TomOSPC_angle(joint2_angle):
    """
    TODO: doc. 
    Input: 
        angle (float):
    Output:
        tomospc_angle_j2 (int): 
    """
    tomospc_angle_j2 = -joint2_angle
    return tomospc_angle_j2

def joint3_angle_to_TomOSPC_angle(joint3_angle):
    """
    TODO: doc. 
    Input: 
        angle (float):
    Output:
        tomospc_angle_j3 (int): 
    """
    tomospc_angle_j3 = -joint3_angle
    return tomospc_angle_j3

def joint4_angle_to_TomOSPC_angle(joint4_angle):
    """
    TODO: doc. 
    Input: 
        angle (float):
    Output:
        tomospc_angle_j4 (int): 
    """
    tomospc_angle_j4 = joint4_angle
    return tomospc_angle_j4

def joint5_angle_to_TomOSPC_angle(joint5_angle):
    """
    TODO: doc. 
    Input: 
        angle (float):
    Output:
        tomospc_angle_j5 (int): 
    """
    if joint5_angle > 115:
        tomospc_angle_j5 = round(min(210, -joint5_angle + 360))
    else:
        tomospc_angle_j5 = round(joint5_angle)

    tomospc_angle_j5 = joint5_angle
    return tomospc_angle_j5

def joint6_angle_to_TomOSPC_angle(joint6_angle):
    """
    TODO: doc. 
    Input: 
        angle (float):
    Output:
        tomospc_angle_j6 (int): 
    """
    tomospc_angle_j6 = joint6_angle
    return tomospc_angle_j6

def calculate_yaw_angle(rotation_matrix, cvt_to_degree=True):
    """
    Calculate the psi angle 
    Input:
        rotation_matrix (np.array): shape = (3, 3), each column is a vector (x, y, z) (horizontally stack)
        cvt_to_degree (bool): whether to convert to degree
    Output:
        yaw_angle:
    """
    assert rotation_matrix.shape == (3, 3)

    yaw_angle = np.arctan2(rotation_matrix[0, 1], rotation_matrix[0, 0])
    if cvt_to_degree:
        yaw_angle = np.degrees(yaw_angle)
    return yaw_angle

def calculate_pitch_angle(rotation_matrix, cvt_to_degree=True):
    """
    Calculate the theta angle 
    Input:
        rotation_matrix (np.array): shape = (3, 3), each column is a vector (x, y, z) (horizontally stack)
        cvt_to_degree (bool): whether to convert to degree
    Output:
        theta_angle:
    """
    assert rotation_matrix.shape == (3, 3)

    theta_angle = -np.arcsin(rotation_matrix[0, -1])
    if cvt_to_degree:
        theta_angle = np.degrees(theta_angle)
    return theta_angle

def get_landmark_vector(XYZ_landmarks, landmark_dictionary, landmark_name):
    """
    TODO: Doc.
    Get vector based on "landmark_name" 
    Input:
        XYZ_landmarks:
        landmark_dictionary:
        landmark_name (Tuple): ("shoulder", "elbow", "wrist")
    Output:
        landmark_vec (np.array): shape = (3,)
    """
    assert landmark_name in ("shoulder", "elbow", "wrist")
    if landmark_name == "shoulder":
        landmark_vec = get_shoulder_vector(XYZ_landmarks, landmark_dictionary)
    elif landmark_name == "elbow":
        landmark_vec = get_elbow_vector(XYZ_landmarks, landmark_dictionary)
    else:
        landmark_vec = get_wrist_vector(XYZ_landmarks, landmark_dictionary)
    return landmark_vec

def get_shoulder_vector(XYZ_landmarks, landmark_dictionary):
    """
    TODO: Doc.
    Get shoulder vector to compute angle j1 and angle j2
    Input:
        XYZ_landmarks:
        landmark_dictionary:
    Output:
        shoulder_vec:
    """
    shoulder_vec = XYZ_landmarks[landmark_dictionary.index("left elbow")].copy()
    return shoulder_vec

def get_elbow_vector(XYZ_landmarks, landmark_dictionary):
    """
    TODO: Doc.
    Get elbow vector to compute angle j3 and angle j4
    Input:
        XYZ_landmarks:
        landmark_dictionary:
    Output:
        elbow_vec:
    """
    wrist_landmark = XYZ_landmarks[landmark_dictionary.index("WRIST")].copy()
    left_elbow_landmark = XYZ_landmarks[landmark_dictionary.index("left elbow")].copy()
    elbow_vec = wrist_landmark - left_elbow_landmark
    return elbow_vec

def get_wrist_vector(XYZ_landmarks, landmark_dictionary):
    """
    TODO: Doc.
    Get wrist vector to compute angle j3 and angle j4
    Input:
        XYZ_landmarks:
        landmark_dictionary:
    Output:
        wrist_vec:
    """
    wrist_landmark = XYZ_landmarks[landmark_dictionary.index("WRIST")].copy()
    index_finger_landmark = XYZ_landmarks[landmark_dictionary.index("INDEX_FINGER_MCP")].copy()
    middle_finger_landmark = XYZ_landmarks[landmark_dictionary.index("MIDDLE_FINGER_MCP")].copy()

    u_wrist = index_finger_landmark - wrist_landmark
    v_wrist = middle_finger_landmark = wrist_landmark

    wrist_vec = np.cross(v_wrist, u_wrist)
    return wrist_vec

def create_rot_mat(unit_vector, joint_name):
    """
    TODO: doc.
    Input:
        unit_vector (np.array): unit_vector with shape = (3,)
        joint_name (Tuple): ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
    Ouptut:
        rot_mat (np.array): shape = (3, 3); each column is a vector (x, y, z) (horizontally stack)
    """
    assert joint_name in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
    assert np.abs(1 - np.linalg.norm(unit_vector)) < 1e-3

    if joint_name == "joint1":
        rot_mat = create_joint1_rot_mat(unit_vector)
    elif joint_name == "joint2":
        rot_mat = create_joint2_rot_mat(unit_vector)
    elif joint_name == "joint3":
        rot_mat = create_joint1_rot_mat(unit_vector)
    elif joint_name == "joint4":
        rot_mat = create_joint2_rot_mat(unit_vector)
    elif joint_name == "joint5":
        rot_mat = create_joint1_rot_mat(unit_vector)
    else:
        rot_mat = create_joint2_rot_mat(unit_vector)
    return rot_mat

def create_joint1_rot_mat(j1_x_unit):
    """
    TODO: doc.
    Joint 1 rotates about z-axis of origin_coordinate. Therefore, j1_z has to be
    aligned with the origin_z
    Input:
        j1_x_unit (np.array): unit_vector with shape = (3,)
    Output:
        j1_rot_mat (np.array): shape = (3, 3); each column is a vector (x, y, z) (horizontally stack)
    """
    j1_y_unit = np.array([-j1_x_unit[1], j1_x_unit[0], j1_x_unit[2]])
    j1_z_unit = np.cross(j1_x_unit, j1_y_unit)

    assert np.abs(1 - np.linalg.norm(j1_y_unit)) < 1e-3
    assert np.abs(1 - np.linalg.norm(j1_z_unit)) < 1e-3

    j1_rot_mat = np.hstack([j1_x_unit[:, None], 
        j1_y_unit[:, None], 
        j1_z_unit[:, None]])

    return j1_rot_mat

def create_joint2_rot_mat(j2_x_unit):
    """
    TODO: doc.
    Joint 2 rotates about y-axis of j1_coordinate. Therefore, j2_y has to be
    aligned with the j1_y
    Input:
        j2_x_unit (np.array): unit_vector with shape = (3,)
    Output:
        j2_rot_mat (np.array): shape = (3, 3); each column is a vector (x, y, z) (horizontally stack)
    """
    j2_z_unit = np.array([-j2_x_unit[-1], 
        j2_x_unit[1], 
        j2_x_unit[0]])
    j2_y_unit = np.cross(j2_z_unit, j2_x_unit)

    assert np.abs(1 - np.linalg.norm(j2_y_unit)) < 1e-3
    assert np.abs(1 - np.linalg.norm(j2_z_unit)) < 1e-3

    j2_rot_mat = np.hstack([j2_x_unit[:, None], 
        j2_y_unit[:, None], 
        j2_z_unit[:, None]])

    return j2_rot_mat

def calculate_the_next_two_joints_angle(XYZ_landmarks,
    landmark_name,
    landmark_dictionary,
    parent_coordinate,
    vector_in_prev_frame,
    rotation_matrix_to_rearrange_coordinate,
    angle_range_of_two_joints,
    axis_to_get_the_opposite_if_angle_exceed_the_limit_of_two_joints,
    get_the_opposite_of_two_joints,
    limit_angle_of_two_joints):
    """
    TODO: Doc.
    The first joint always rotates about the z-axis w.r.t. the parent_coordinate (yaw_angle). 
    The second joint always rotates about the y-axis w.r.t the above joint coordinate (pitch_angle).
    """
    assert landmark_name in ("shoulder", "elbow", "wrist")
    if landmark_name == "shoulder":
        joint_name = ("joint1", "joint2")
    elif landmark_name == "elbow":
        joint_name = ("joint3", "joint4")
    else:
        joint_name = ("joint5", "joint6")

    older_brother_result = get_angle_and_rot_mats(
        XYZ_landmarks=XYZ_landmarks,
        landmark_name=landmark_name,
        landmark_dictionary=landmark_dictionary,
        parent_rot_mat=parent_coordinate,
        vector_in_prev_frame=vector_in_prev_frame,
        calculate_angle_func=calculate_yaw_angle,
        axis_to_remove="z",
        rotation_matrix=rotation_matrix_to_rearrange_coordinate,
        joint_name=joint_name[0],
        angle_range=angle_range_of_two_joints[0],
        axis_to_get_the_opposite_if_angle_exceed_the_limit=axis_to_get_the_opposite_if_angle_exceed_the_limit_of_two_joints[0],
        get_the_opposite=get_the_opposite_of_two_joints[0],
        limit_angle=limit_angle_of_two_joints[0])
    older_brother_angle = older_brother_result["angle"]
    older_brother_rot_mat_wrt_origin = older_brother_result["rot_mat_wrt_origin"]
    older_brother_rot_mat_wrt_parent = older_brother_result["rot_mat_wrt_parent"]
    is_older_brother_vector_near_zero_after_proj = older_brother_result["is_near_zero_after_project"]

    younger_brother_result = get_angle_and_rot_mats(
        XYZ_landmarks=XYZ_landmarks,
        landmark_name=landmark_name,
        landmark_dictionary=landmark_dictionary,
        parent_rot_mat=older_brother_rot_mat_wrt_origin,
        vector_in_prev_frame=vector_in_prev_frame,
        calculate_angle_func=calculate_pitch_angle,
        axis_to_remove="y",
        rotation_matrix=None,
        joint_name=joint_name[1],
        angle_range=angle_range_of_two_joints[1],
        axis_to_get_the_opposite_if_angle_exceed_the_limit=axis_to_get_the_opposite_if_angle_exceed_the_limit_of_two_joints[1],
        get_the_opposite=get_the_opposite_of_two_joints[1],
        limit_angle=limit_angle_of_two_joints[1])
    younger_brother_angle = younger_brother_result["angle"]
    younger_brother_rot_mat_wrt_origin = younger_brother_result["rot_mat_wrt_origin"]
    younger_brother_rot_mat_wrt_older_brother = younger_brother_result["rot_mat_wrt_parent"]
    is_younger_brother_vector_near_zero_after_proj = younger_brother_result["is_near_zero_after_project"]

    if (not is_older_brother_vector_near_zero_after_proj and
        not is_younger_brother_vector_near_zero_after_proj):
        vector_in_current_frame = get_landmark_vector(XYZ_landmarks, landmark_dictionary, landmark_name)
    else:
        vector_in_current_frame = vector_in_prev_frame.copy()

    results = {
        "older_brother_angle": older_brother_angle,
        "older_brother_rot_mat_wrt_origin": older_brother_rot_mat_wrt_origin,
        "older_brother_rot_mat_wrt_parent": older_brother_rot_mat_wrt_parent,
        "younger_brother_angle": younger_brother_angle,
        "younger_brother_rot_mat_wrt_origin": younger_brother_rot_mat_wrt_origin,
        "younger_brother_rot_mat_wrt_older_brother": younger_brother_rot_mat_wrt_older_brother,
        "vector_in_current_frame": vector_in_current_frame
    }

    return results

def calculate_six_arm_angles(XYZ_landmarks, origin_coordinate, landmark_dictionary):
    global shoulder_vector_in_prev_frame
    global elbow_vector_in_prev_frame
    global wrist_vector_in_prev_frame
    global rotation_matrix_for_shoulder 
    global rotation_matrix_for_elbow 
    global rotation_matrix_for_wrist

    j1_j2_results = calculate_the_next_two_joints_angle(
        XYZ_landmarks=XYZ_landmarks,
        landmark_dictionary=landmark_dictionary,
        landmark_name="shoulder",
        parent_coordinate=origin_coordinate,
        vector_in_prev_frame=shoulder_vector_in_prev_frame,
        rotation_matrix_to_rearrange_coordinate=rotation_matrix_for_shoulder,
        angle_range_of_two_joints=[[-195, 86], [None, None]],
        axis_to_get_the_opposite_if_angle_exceed_the_limit_of_two_joints=["y", None],
        get_the_opposite_of_two_joints=[True, True],
        limit_angle_of_two_joints=[True, False]
    )

    angle_j1 = j1_j2_results["older_brother_angle"]
    j1_rot_mat_wrt_origin = j1_j2_results["older_brother_rot_mat_wrt_origin"]
    angle_j2 = j1_j2_results["younger_brother_angle"]
    j2_rot_mat_wrt_origin = j1_j2_results["younger_brother_rot_mat_wrt_origin"]
    j2_rot_mat_wrt_j1 = j1_j2_results["younger_brother_rot_mat_wrt_older_brother"] 
    shoulder_vector_in_prev_frame = j1_j2_results["vector_in_current_frame"].copy()

    j3_j4_results = calculate_the_next_two_joints_angle(
        XYZ_landmarks=XYZ_landmarks,
        landmark_dictionary=landmark_dictionary,
        landmark_name="elbow",
        parent_coordinate=j2_rot_mat_wrt_origin,
        vector_in_prev_frame=elbow_vector_in_prev_frame,
        rotation_matrix_to_rearrange_coordinate=rotation_matrix_for_elbow,
        angle_range_of_two_joints=[[None, None], [None, None]],
        axis_to_get_the_opposite_if_angle_exceed_the_limit_of_two_joints=[None, None],
        get_the_opposite_of_two_joints=[True, True],
        limit_angle_of_two_joints=[False, False]
    )

    angle_j3 = j3_j4_results["older_brother_angle"]
    j3_rot_mat_wrt_origin = j3_j4_results["older_brother_rot_mat_wrt_origin"]
    j3_rot_mat_wrt_j2 = j3_j4_results["older_brother_rot_mat_wrt_parent"]
    angle_j4 = j3_j4_results["younger_brother_angle"]
    j4_rot_mat_wrt_origin = j3_j4_results["younger_brother_rot_mat_wrt_origin"]
    j4_rot_mat_wrt_j3 = j3_j4_results["younger_brother_rot_mat_wrt_older_brother"] 
    elbow_vector_in_prev_frame = j3_j4_results["vector_in_current_frame"].copy()

    j5_j6_resulst = calculate_the_next_two_joints_angle(
        XYZ_landmarks=XYZ_landmarks,
        landmark_dictionary=landmark_dictionary,
        landmark_name="wrist",
        parent_coordinate=j4_rot_mat_wrt_origin,
        vector_in_prev_frame=wrist_vector_in_prev_frame,
        rotation_matrix_to_rearrange_coordinate=rotation_matrix_for_wrist,
        angle_range_of_two_joints=[[None, None], [None, None]],
        axis_to_get_the_opposite_if_angle_exceed_the_limit_of_two_joints=[None, None],
        get_the_opposite_of_two_joints=[True, True],
        limit_angle_of_two_joints=[False, False]
    )

    angle_j5 = j5_j6_resulst["older_brother_angle"]
    j5_rot_mat_wrt_origin = j5_j6_resulst["older_brother_rot_mat_wrt_origin"]
    j5_rot_mat_wrt_j4 = j5_j6_resulst["older_brother_rot_mat_wrt_parent"]
    angle_j6 = j5_j6_resulst["younger_brother_angle"]
    j6_rot_mat_wrt_origin = j5_j6_resulst["younger_brother_rot_mat_wrt_origin"]
    j6_rot_mat_wrt_j5 = j5_j6_resulst["younger_brother_rot_mat_wrt_older_brother"]
    wrist_vector_in_prev_frame = j5_j6_resulst["vector_in_current_frame"].copy()

    angles = (angle_j1, angle_j2, angle_j3, angle_j4, angle_j5, angle_j6)
    coordinates_wrt_origin = (j1_rot_mat_wrt_origin, j2_rot_mat_wrt_origin,
        j3_rot_mat_wrt_origin, j4_rot_mat_wrt_origin,
        j5_rot_mat_wrt_origin, j6_rot_mat_wrt_origin)
    coordinates_wrt_its_parent = (j1_rot_mat_wrt_origin, j2_rot_mat_wrt_j1,
        j3_rot_mat_wrt_j2, j4_rot_mat_wrt_j3,
        j5_rot_mat_wrt_j4, j6_rot_mat_wrt_j5)

    return angles, coordinates_wrt_origin, coordinates_wrt_its_parent
