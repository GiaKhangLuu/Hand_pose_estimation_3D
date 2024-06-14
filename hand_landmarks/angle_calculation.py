import numpy as np
import math
from numpy.typing import NDArray

def project_to(mat, plane=None):
    """
    project_to = None means keep the [x, y, z]
    """

    mask = [1, 1, 1]
    if plane is not None:
        assert plane in ["xy", "xz", "yz"]
        if plane == "xy":
            mask = [1, 1, 0]
        elif plane == "xz":
            mask = [1, 0, 1]
        else:
            mask = [0, 1, 1]
    mat = mat * mask
    return mat

def calc_angle_between(a, b):
    """
    Cal. the angle between two vector using dot_product.
    """

    assert a.shape[-1] == b.shape[-1]

    dot_product = np.sum(a * b, axis=1)  # calculate dot product by element-wise style instead of using np.dot
    magnitude_a = np.linalg.norm(a, axis=1)
    magnitude_b = np.linalg.norm(b, axis=1)
    cos_theta = dot_product / (magnitude_a * magnitude_b)
    angle_radians = np.arccos(cos_theta)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

def get_direction(a, b, reference_dim=0):
    """
    The order of `a`, `b` is important, because
    this func. calculate first calculate
    the cross_product (`c`) of `a` and `b`. Then
    get the sign of `c[reference_dim]`.
    """

    c = np.cross(a, b)
    if c.ndim == 1:
        c = c[None, :]
    ref_values = c[:, reference_dim] + 1e-9 # avoid dividing by 0
    signs = ref_values / np.absolute(ref_values) 
    return signs 

def calculate_angles_of_thumb(fingers_XYZ_wrt_wrist: NDArray) -> NDArray:
    x_unit = np.array([1, 0, 0])
    y_unit = np.array([0, 1, 0])
    z_unit = np.array([0, 0, 1])

    thumb = fingers_XYZ_wrt_wrist[0, ...]

    a = np.array([x_unit, 
                  y_unit, 
                  thumb[1, ...] - thumb[0, ...]])
    b = np.array([project_to(thumb[2, ...] - thumb[0, ...], plane="xz"),
                  project_to(thumb[3, ...] - thumb[1, ...], plane="yz"),
                  thumb[2, ...] - thumb[1, ...]])

    thumb_angles = calc_angle_between(a, b)                  

    # According to the right thumb: a[0, :] is x, b[0, :] is y => j_1_sign = '-' while J11 (of the left hand) move from middle finger 
    # to the index finger and j_1_sign = '+' while J11 move from the middle finger to the last finger (pinky finger)
    j_1_sign = get_direction(a[0, :], b[0, :], 1)[0] 
    # According to the right thumb: a[1, :] is x, b[1, :] is y => j_2_sign = '+' while J12 (of the left hand) move from middle finger
    # to the index finger and j_2_sign = '- while J12 move from the middle finger to the last finger (pinky finger)
    j_2_sign = get_direction(a[1, :], b[1, :], 0)[0] 
    j_3_sign = get_direction(a[-1, :], b[-1, :], -1)[0]
    thumb_angles = thumb_angles * [j_1_sign, j_2_sign, j_3_sign]

    # Joint 4 uses the same value of joint 3
    thumb_angles = np.concatenate([thumb_angles, thumb_angles[-1:]])

    return thumb_angles

def get_angles_of_hand_joints(wrist_XYZ: NDArray, fingers_XYZ_wrt_wrist: NDArray, degrees=False) -> NDArray:

    assert np.sum(np.abs(wrist_XYZ)).astype(np.int8) == 0

    angles = np.zeros(shape=fingers_XYZ_wrt_wrist.shape[:-1])
    y_unit = np.array([0, 1, 0])

    # Angles of J21 - > J51
    # The order of params a and b is important here, because we will compute cross_prod to get the direction
    a = y_unit
    b = fingers_XYZ_wrt_wrist[1:, 1, :] - fingers_XYZ_wrt_wrist[1:, 0, :]
    a = np.full_like(b, a)
    b = project_to(b, plane="yz")
    angles_joint_i_1 = calc_angle_between(a, b)
    signs = get_direction(a, b, reference_dim=0)  # a and b lie on "yz", so c lines on "x" => "reference_dim" is 0 (x)
    angles_joint_i_1 = angles_joint_i_1 * signs
    angles[1:, 0] = angles_joint_i_1

    # Angles of J22 - > J52
    # The order of params a and b is important here, because we will compute determinant to get the direction
    a = fingers_XYZ_wrt_wrist[1:, 0, :]
    b = fingers_XYZ_wrt_wrist[1:, 1, :] - fingers_XYZ_wrt_wrist[1:, 0, :]
    b = project_to(b, plane="xy")
    angles_joint_i_2 = calc_angle_between(a, b)
    signs = get_direction(a, b, reference_dim=-1)
    angles_joint_i_2 = angles_joint_i_2 * signs
    angles[1:, 1] = angles_joint_i_2

    # Angles of J23 - > J53
    # The order of params a and b is important here, because we will compute determinant to get the direction
    a = fingers_XYZ_wrt_wrist[1:, 1, :] - fingers_XYZ_wrt_wrist[1:, 0, :] 
    b = fingers_XYZ_wrt_wrist[1:, 2, :] - fingers_XYZ_wrt_wrist[1:, 1, :]
    b = project_to(b, plane="xy")
    angles_joint_i_3 = calc_angle_between(a, b)
    signs = get_direction(a, b, reference_dim=-1)
    angles_joint_i_3 = angles_joint_i_3 * signs
    angles[1:, 2] = angles_joint_i_3

    # Angles of 214 - > J54
    # The order of params a and b is important here, because we will compute determinant to get the direction
    a = fingers_XYZ_wrt_wrist[1:, 2, :] - fingers_XYZ_wrt_wrist[1:, 1, :]
    b = fingers_XYZ_wrt_wrist[1:, 3, :] - fingers_XYZ_wrt_wrist[1:, 2, :]
    b = project_to(b, plane="xy")
    angles_joints_i_4 = calc_angle_between(a, b)
    signs = get_direction(a, b, reference_dim=-1)
    angles_joints_i_4 = angles_joints_i_4 * signs
    angles[1:, 3] = angles_joints_i_4

    # Apply weight for joint 1
    joint_1_weight = np.interp(np.absolute(angles[1:, 1]), [0, 90], [1, 0])
    angles[1:, 0] *= joint_1_weight

    angles[0, :] = calculate_angles_of_thumb(fingers_XYZ_wrt_wrist)

    # Temporarily hard code the mapping angle
    angles[0, 0] = map_real_life_angles_to_rviz_angles(angles[0, 0], [-70, 0], [-35, 35])
    angles[0, 1] = map_real_life_angles_to_rviz_angles(angles[0, 1], [0, 40], [-90, 8])
    angles[0, 2] = map_real_life_angles_to_rviz_angles(angles[0, 2], [-50, -10], [-90, 0])
    angles[0, 3] = map_real_life_angles_to_rviz_angles(angles[0, 3], [-50, -10], [-90, 0])

    angles[:, 0] = bound_angles(angles[:, 0], -35, 35)
    angles[:, 1] = bound_angles(angles[:, 1], -100, 8)
    angles[:, 2] = bound_angles(angles[:, 2], -100, 8)
    angles[:, 3] = bound_angles(angles[:, 3], -100, 8)

    angles = np.round(angles)

    if not degrees:
        angles = degree_2_rad(angles)
    return angles

def degree_2_rad(angles):
    radian_angles = angles * math.pi / 180                                     
    return radian_angles

def map_real_life_angles_to_rviz_angles(angles, xp, fp):
    angles = np.interp(angles, xp, fp)
    return angles

def bound_angles(angles, min_degree, max_degree):
    angles = np.clip(angles, min_degree, max_degree)
    return angles
