"""
Currently support for left arm only
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
from angle_calculation_utilities import calculate_the_next_two_joints_angle

shoulder_vector_in_prev_frame = None
elbow_vector_in_prev_frame = None
wrist_vector_in_prev_frame = None

rotation_matrix_for_shoulder = np.eye(3)
rotation_matrix_for_elbow = R.from_euler("xz", [90, -90], degrees=True).as_matrix()
rotation_matrix_for_wrist = R.from_euler("y", -90, degrees=True).as_matrix()

bound = 2

joint1_max = 86 - bound
joint1_min = -195 + bound

joint2_max = 92 - bound
joint2_min = -3 + bound

joint3_max = 143 - bound
joint3_min = -143 + bound

joint4_max = 22 - bound
joint4_min = -91 + bound

joint5_max = 206 - bound
joint5_min = -115 + bound

joint6_max = 52 - bound
joint6_min = -69 + bound

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
    Get wrist vector to compute angle j5 and angle j6
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
    v_wrist = middle_finger_landmark - wrist_landmark

    wrist_vec = np.cross(v_wrist, u_wrist)
    return wrist_vec

def joint1_angle_to_TomOSPC_angle(joint1_angle):
    """
    TODO: doc. 
    Input: 
        angle (float):
    Output:
        tomospc_angle_j1 (int): 
    """
    if -90 <= joint1_angle <= 180:
        tomospc_angle_j1 = -joint1_angle
    else:
        tomospc_angle_j1 = -joint1_angle - 360
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
    global joint5_max
    if joint5_angle > 115:
        tomospc_angle_j5 = min(joint5_max, -joint5_angle + 360)
    else:
        tomospc_angle_j5 = -joint5_angle

    return tomospc_angle_j5

def joint6_angle_to_TomOSPC_angle(joint6_angle):
    """
    TODO: doc. 
    Input: 
        angle (float):
    Output:
        tomospc_angle_j6 (int): 
    """
    tomospc_angle_j6 = -joint6_angle
    return tomospc_angle_j6

def calculate_six_arm_angles(XYZ_landmarks, origin_coordinate, landmark_dictionary):
    """
    To calculate a couple angles for each joint, we need THREE values:
        1. A vector which acts as an x-vector in order to create a
            child coordinate.
        2. A rotation matrices which helps to rearrange the x-axis, 
            y-axis and z-axis into the new ones. This new coordinate
            ensures that the first joint rotates about the z-axis
            and the second joint rotates about the y-axis.
        3. Two mapping function. Each angle differs from an angles
            in robot, therefore, we have to transform the angle of
            real person into the angle of robot.
    """

    global shoulder_vector_in_prev_frame
    global elbow_vector_in_prev_frame
    global wrist_vector_in_prev_frame
    global rotation_matrix_for_shoulder 
    global rotation_matrix_for_elbow 
    global rotation_matrix_for_wrist
    global joint1_max 
    global joint1_min 
    global joint2_max 
    global joint2_min 
    global joint3_max 
    global joint3_min 
    global joint4_max 
    global joint4_min 
    global joint5_max 
    global joint5_min 
    global joint6_max 
    global joint6_min 

    landmarks_name = ("shoulder", "elbow", "wrist")
    joint_name = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
    vec_landmark_container = []
    mapping_angle_function_containter = [
        (joint1_angle_to_TomOSPC_angle, joint2_angle_to_TomOSPC_angle),
        (joint3_angle_to_TomOSPC_angle, joint4_angle_to_TomOSPC_angle),
        (joint5_angle_to_TomOSPC_angle, joint6_angle_to_TomOSPC_angle)
    ]
    rot_mat_to_rearrange_container = [
        rotation_matrix_for_shoulder,
        rotation_matrix_for_elbow,
        rotation_matrix_for_wrist
    ]

    for i, name in enumerate(landmarks_name):
        vec_landmark = get_landmark_vector(XYZ_landmarks,
            landmark_dictionary, name)
        vec_landmark_container.append(vec_landmark)

    j1_j2_results = calculate_the_next_two_joints_angle(
        vector_landmark=vec_landmark_container[0],
        map_to_robot_angle_funcs=mapping_angle_function_containter[0],
        parent_coordinate=origin_coordinate,
        vector_in_prev_frame=shoulder_vector_in_prev_frame,
        rotation_matrix_to_rearrange_coordinate=rot_mat_to_rearrange_container[0],
        angle_range_of_two_joints=[[joint1_min, joint1_max], [joint2_min, joint2_max]],
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
        vector_landmark=vec_landmark_container[1],
        map_to_robot_angle_funcs=mapping_angle_function_containter[1],
        parent_coordinate=j2_rot_mat_wrt_origin,
        vector_in_prev_frame=elbow_vector_in_prev_frame,
        rotation_matrix_to_rearrange_coordinate=rot_mat_to_rearrange_container[1],
        angle_range_of_two_joints=[[joint3_min, joint3_max], [joint4_min, joint4_max]],
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
        vector_landmark=vec_landmark_container[2],
        map_to_robot_angle_funcs=mapping_angle_function_containter[2],
        parent_coordinate=j4_rot_mat_wrt_origin,
        vector_in_prev_frame=wrist_vector_in_prev_frame,
        rotation_matrix_to_rearrange_coordinate=rot_mat_to_rearrange_container[2],
        angle_range_of_two_joints=[[joint5_min, joint5_max], [joint6_min, joint6_max]],
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
