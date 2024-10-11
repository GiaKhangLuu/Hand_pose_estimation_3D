import numpy as np
from scipy.spatial.transform import Rotation as R

length_threshold = 0.03
axis_names = ['x', 'y', 'z']

def get_angle_and_rot_mats(vector_landmark, 
    create_child_coordinate_func,
    calculate_angle_func,
    map_to_robot_angle_func,
    parent_rot_mat,
    vector_in_prev_frame,
    axis_to_remove,
    rotation_matrix=None,
    axis_to_get_the_opposite_if_angle_exceed_the_limit=None,
    angle_range=None,
    get_the_opposite=False,
    limit_angle=False,
    clip_angle=True):
    """
    TODO: Doc.
    """
    global length_threshold
    global axis_names
    vector_in_current_frame_is_near_zero_after_project = False
    if rotation_matrix is not None:
        parent_rot_mat = parent_rot_mat @ rotation_matrix
    parent_rot_mat_inv = parent_rot_mat.T
    vec_landmark = vector_landmark.copy()
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
    rot_mat_wrt_parent = create_child_coordinate_func(unit_vec_landmark_wrt_parent)
    angle = calculate_angle_func(rot_mat_wrt_parent)
    angle = map_to_robot_angle_func(angle)

    if limit_angle:
        assert angle_range is not None
        assert axis_to_get_the_opposite_if_angle_exceed_the_limit is not None
        if angle > angle_range[1] or angle < angle_range[0]:
            axis_idx = axis_names.index(axis_to_get_the_opposite_if_angle_exceed_the_limit)
            unit_vec_landmark_wrt_parent[axis_idx] = -unit_vec_landmark_wrt_parent[axis_idx]
        rot_mat_wrt_parent = create_child_coordinate_func(unit_vec_landmark_wrt_parent)
        angle = calculate_angle_func(rot_mat_wrt_parent)
        angle = map_to_robot_angle_func(angle)

    if clip_angle:
        angle = np.clip(angle, angle_range[0], angle_range[1])

    rot_mat_wrt_origin = np.matmul(parent_rot_mat, rot_mat_wrt_parent)

    result = {
        "angle": angle,
        "rot_mat_wrt_origin": rot_mat_wrt_origin,
        "rot_mat_wrt_parent": rot_mat_wrt_parent,
        "is_near_zero_after_project": vector_in_current_frame_is_near_zero_after_project
    }

    return result

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

def create_coordinate_rotate_about_z(x_unit):
    """
    The new coordinate, which we are going to create, rotates 
    about z-axis of the parent coordinate. Therefore, the new
    z-axis has to be aligned with the z-axis of the parent.
    Input:
        x_unit (np.array): unit_vector with shape = (3,)
    Output:
        rot_mat (np.array): shape = (3, 3); each column is a vector (x, y, z) (horizontally stack)
    """
    assert np.abs(1 - np.linalg.norm(x_unit)) < 1e-3

    y_unit = np.array([-x_unit[1], x_unit[0], x_unit[2]])
    z_unit = np.cross(x_unit, y_unit)

    assert np.abs(1 - np.linalg.norm(y_unit)) < 1e-3
    assert np.abs(1 - np.linalg.norm(z_unit)) < 1e-3

    rot_mat = np.hstack([x_unit[:, None], 
        y_unit[:, None], 
        z_unit[:, None]])

    return rot_mat

def create_coordinate_rotate_about_y(x_unit):
    """
    The new coordinate, which we are going to create, rotates
    about y-axis of the parent coordinate. Therefore, the new
    y-axis has to be aligned with the y-axis of the parent.
    Input:
        x_unit (np.array): unit_vector with shape = (3,)
    Output:
        rot_mat (np.array): shape = (3, 3); each column is a vector (x, y, z) (horizontally stack)
    """
    assert np.abs(1 - np.linalg.norm(x_unit)) < 1e-3

    z_unit = np.array([-x_unit[-1], 
        x_unit[1], 
        x_unit[0]])
    y_unit = np.cross(z_unit, x_unit)

    assert np.abs(1 - np.linalg.norm(y_unit)) < 1e-3
    assert np.abs(1 - np.linalg.norm(z_unit)) < 1e-3

    rot_mat = np.hstack([x_unit[:, None], 
        y_unit[:, None], 
        z_unit[:, None]])

    return rot_mat

def calculate_the_next_two_joints_angle(vector_landmark,
    map_to_robot_angle_funcs,
    parent_coordinate,
    vector_in_prev_frame,
    rotation_matrix_to_rearrange_coordinate,
    angle_range_of_two_joints,
    axis_to_get_the_opposite_if_angle_exceed_the_limit_of_two_joints,
    get_the_opposite_of_two_joints,
    limit_angle_of_two_joints,
    clip_angle_of_two_joints,
    calculate_the_second_joint=True):
    """
    TODO: Doc.
    The first joint always rotates about the z-axis w.r.t. the parent_coordinate (yaw_angle). 
    The second joint always rotates about the y-axis w.r.t the above joint coordinate (pitch_angle).
    """

    mapping_angle_first_joint, mapping_angle_second_joint = map_to_robot_angle_funcs
    min_max_first_joint, min_max_second_joint = angle_range_of_two_joints
    axis_to_get_oppo_if_exceed_first_joint, axis_to_get_oppo_if_exceed_second_joint = axis_to_get_the_opposite_if_angle_exceed_the_limit_of_two_joints
    get_oppo_flag_first_joint, get_oppo_flag_second_joint = get_the_opposite_of_two_joints
    limit_flag_first_joint, limit_flag_second_joint = limit_angle_of_two_joints
    clip_flag_first_joint, clip_flag_second_joint = clip_angle_of_two_joints

    older_brother_result = get_angle_and_rot_mats(
        vector_landmark=vector_landmark,
        create_child_coordinate_func=create_coordinate_rotate_about_z,
        calculate_angle_func=calculate_yaw_angle,
        map_to_robot_angle_func=mapping_angle_first_joint,
        parent_rot_mat=parent_coordinate,
        vector_in_prev_frame=vector_in_prev_frame,
        axis_to_remove="z",
        rotation_matrix=rotation_matrix_to_rearrange_coordinate,
        axis_to_get_the_opposite_if_angle_exceed_the_limit=axis_to_get_oppo_if_exceed_first_joint,
        angle_range=min_max_first_joint,
        get_the_opposite=get_oppo_flag_first_joint,
        limit_angle=limit_flag_first_joint,
        clip_angle=clip_flag_first_joint)
    older_brother_angle = older_brother_result["angle"]
    older_brother_rot_mat_wrt_origin = older_brother_result["rot_mat_wrt_origin"]
    older_brother_rot_mat_wrt_parent = older_brother_result["rot_mat_wrt_parent"]
    is_older_brother_vector_near_zero_after_proj = older_brother_result["is_near_zero_after_project"]

    if calculate_the_second_joint:
        younger_brother_result = get_angle_and_rot_mats(
            vector_landmark=vector_landmark,
            create_child_coordinate_func=create_coordinate_rotate_about_y,
            calculate_angle_func=calculate_pitch_angle,
            map_to_robot_angle_func=mapping_angle_second_joint,
            parent_rot_mat=older_brother_rot_mat_wrt_origin,
            vector_in_prev_frame=vector_in_prev_frame,
            axis_to_remove="y",
            rotation_matrix=None,
            axis_to_get_the_opposite_if_angle_exceed_the_limit=axis_to_get_oppo_if_exceed_second_joint,
            angle_range=min_max_second_joint,
            get_the_opposite=get_oppo_flag_second_joint,
            limit_angle=limit_flag_second_joint,
            clip_angle=clip_flag_second_joint)
        younger_brother_angle = younger_brother_result["angle"]
        younger_brother_rot_mat_wrt_origin = younger_brother_result["rot_mat_wrt_origin"]
        younger_brother_rot_mat_wrt_older_brother = younger_brother_result["rot_mat_wrt_parent"]
        is_younger_brother_vector_near_zero_after_proj = younger_brother_result["is_near_zero_after_project"]
    else:
        younger_brother_angle = None
        younger_brother_rot_mat_wrt_origin = None
        younger_brother_rot_mat_wrt_older_brother = None
        is_younger_brother_vector_near_zero_after_proj = False


    if (not is_older_brother_vector_near_zero_after_proj and
        not is_younger_brother_vector_near_zero_after_proj):
        vector_in_current_frame = vector_landmark.copy()
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

