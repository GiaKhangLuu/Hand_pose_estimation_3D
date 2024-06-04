import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.typing import NDArray
from typing import Tuple
from functools import partial
from scipy.spatial.distance import euclidean
from scipy.optimize import minimize

finger_joints_names = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]

def get_xyZ(landmarks, depth, frame_size, sliding_window_size):
    assert depth is not None
    xyz = []
    for landmark in landmarks.landmark:
        x = landmark.x
        y = landmark.y
        z = landmark.z
        xyz.append([x, y, z])
    xyz = np.array(xyz)
    xy_unnorm = unnormalize(xyz, frame_size)
    Z = get_depth(xy_unnorm, depth, sliding_window_size)
    xyZ = np.concatenate([xy_unnorm, Z[:, None]], axis=-1)
    return xyZ

def detect_hand_landmarks(color_img, detector, landmarks_queue, mp_drawing, mp_hands):
    color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

    results = detector.process(color_img)
    if results.multi_hand_landmarks:
        for hand_num, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(color_img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks_queue.put(hand_landmarks)
            if landmarks_queue.qsize() > 1:
                landmarks_queue.get()

    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    return color_img

def _load_config(self):  
    with open(os.path.join(cfg_dir, "right_to_opposite_correctmat.yaml"), 'r') as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
        right_to_opposite_correctmat = configs["right_to_opposite_correctmat"]
    file.close()
    return right_to_opposite_correctmat

def filter_depth(depth_array: NDArray, sliding_window_size, sigma_color, sigma_space) -> NDArray:
    depth_array = depth_array.astype(np.float32)
    depth_array = cv2.bilateralFilter(depth_array, sliding_window_size, sigma_color, sigma_space)
    return depth_array

def unnormalize(arr: NDArray, frame_size) -> NDArray:
    arr = arr[:, :-1]
    arr[:, 0] = arr[:, 0] * frame_size[0]
    arr[:, 1] = arr[:, 1] * frame_size[1]

    """
    Note: some landmarks have value > or < window_height and window_width,
        so this will cause the depth_image out of bound. For now, we just
        clip in in the range of window's dimension. But those values
        properly be removed from the list.
    """

    arr[:, 0] = np.clip(arr[:, 0], 0, frame_size[0] - 1)
    arr[:, 1] = np.clip(arr[:, 1], 0, frame_size[1] - 1)
    return arr

def get_depth(positions: NDArray, depth: NDArray, sliding_window_size) -> NDArray:
    half_size = sliding_window_size // 2
    positions = positions.astype(np.int32)

    x_min = np.maximum(0, positions[:, 0] - half_size)
    x_max = np.minimum(depth.shape[1] - 1, positions[:, 0] + half_size)
    y_min = np.maximum(0, positions[:, 1] - half_size)
    y_max = np.minimum(depth.shape[0] - 1, positions[:, 1] + half_size)

    xy_windows = np.concatenate([x_min[:, None], x_max[:, None], y_min[:, None], y_max[:, None]], axis=-1)

    z_landmarks = []
    for i in range(xy_windows.shape[0]):
        z_values = depth[xy_windows[i, 2]:xy_windows[i, 3] + 1, xy_windows[i, 0]:xy_windows[i, 1] + 1]
        mask = z_values > 0
        z_values = z_values[mask]
        z_median = np.median(z_values)
        z_landmarks.append(z_median)

    return np.array(z_landmarks)

def distance(Z, right_side_xyZ, opposite_xyZ, 
             right_side_cam_intrinsic, 
             opposite_cam_intrinsic,
             right_to_opposite_correctmat):
    right_side_Z, opposite_Z = Z
    right_side_XYZ = np.zeros_like(right_side_xyZ)
    opposite_XYZ = np.zeros_like(opposite_xyZ)

    right_side_XYZ[0] = (right_side_xyZ[0] - right_side_cam_intrinsic[0, -1]) * right_side_Z / right_side_cam_intrinsic[0, 0]
    right_side_XYZ[1] = (right_side_xyZ[1] - right_side_cam_intrinsic[1, -1]) * right_side_Z / right_side_cam_intrinsic[1, 1]
    right_side_XYZ[-1] = right_side_Z

    opposite_XYZ[0] = (opposite_xyZ[0] - opposite_cam_intrinsic[0, -1]) * opposite_Z / opposite_cam_intrinsic[0, 0]
    opposite_XYZ[1] = (opposite_xyZ[1] - opposite_cam_intrinsic[1, -1]) * opposite_Z / opposite_cam_intrinsic[1, 1]
    opposite_XYZ[-1] = opposite_Z

    #homo = np.ones(shape=oak_XYZ.shape[0])
    right_side_XYZ_homo = np.concatenate([right_side_XYZ, [1]])
    right_side_XYZ_in_opposite = np.matmul(right_to_opposite_correctmat, right_side_XYZ_homo.T)
    right_side_XYZ_in_opposite = right_side_XYZ_in_opposite[:-1]
    return euclidean(right_side_XYZ_in_opposite, opposite_XYZ)

def fuse_landmarks_from_two_cameras(opposite_xyZ: NDArray, 
                                    right_side_xyZ: NDArray,
                                    right_side_cam_intrinsic,
                                    opposite_cam_intrinsic,
                                    right_to_opposite_correctmat) -> NDArray:
    right_side_new_Z, opposite_new_Z = [], []
    for i in range(right_side_xyZ.shape[0]):
        right_side_i_xyZ, opposite_i_xyZ = right_side_xyZ[i], opposite_xyZ[i]

        min_dis = partial(distance, right_side_xyZ=right_side_i_xyZ, opposite_xyZ=opposite_i_xyZ,
                          right_side_cam_intrinsic=right_side_cam_intrinsic,
                          opposite_cam_intrinsic=opposite_cam_intrinsic,
                          right_to_opposite_correctmat=right_to_opposite_correctmat)
        result = minimize(min_dis, x0=[right_side_i_xyZ[-1], opposite_i_xyZ[-1]], tol=1e-1)
        right_side_i_new_Z, opposite_i_new_Z = result.x
        right_side_new_Z.append(right_side_i_new_Z)
        opposite_new_Z.append(opposite_i_new_Z)

    right_side_new_xyZ = right_side_xyZ.copy()
    opposite_new_xyZ = opposite_xyZ.copy()

    right_side_new_xyZ[:, -1] = right_side_new_Z
    opposite_new_xyZ[:, -1] = opposite_new_Z
    right_side_new_XYZ = np.zeros_like(right_side_new_xyZ)
    opposite_new_XYZ = np.zeros_like(opposite_new_xyZ)
    right_side_new_XYZ[:, 0] = (right_side_new_xyZ[:, 0] - right_side_cam_intrinsic[0, -1]) * right_side_new_xyZ[:, -1] / right_side_cam_intrinsic[0, 0]
    right_side_new_XYZ[:, 1] = (right_side_new_xyZ[:, 1] - right_side_cam_intrinsic[1, -1]) * right_side_new_xyZ[:, -1] / right_side_cam_intrinsic[1, 1]
    right_side_new_XYZ[:, -1] = right_side_new_xyZ[:, -1]
    opposite_new_XYZ[:, 0] = (opposite_new_xyZ[:, 0] - opposite_cam_intrinsic[0, -1]) * opposite_new_xyZ[:, -1] / opposite_cam_intrinsic[0, 0]
    opposite_new_XYZ[:, 1] = (opposite_new_xyZ[:, 1] - opposite_cam_intrinsic[1, -1]) * opposite_new_xyZ[:, -1] / opposite_cam_intrinsic[1, 1]
    opposite_new_XYZ[:, -1] = opposite_new_xyZ[:, -1]
    fused_landmarks = (right_side_new_XYZ + opposite_new_XYZ) / 2
    return fused_landmarks

def convert_to_wrist_coord(XYZ_landmarks: NDArray) -> Tuple[NDArray, NDArray]:
    u = XYZ_landmarks[finger_joints_names.index("INDEX_FINGER_MCP"), :] - XYZ_landmarks[finger_joints_names.index("WRIST"), :]
    y = XYZ_landmarks[finger_joints_names.index("MIDDLE_FINGER_MCP"), :] - XYZ_landmarks[finger_joints_names.index("WRIST"), :]

    x = np.cross(y, u)
    z = np.cross(x, y)

    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    z = z / np.linalg.norm(z)

    w_c = XYZ_landmarks[finger_joints_names.index("WRIST"), :]

    R = np.array([x, y, z, w_c])
    R = np.concatenate([R, np.expand_dims([0, 0, 0, 1], 1)], axis=1)
    R = np.transpose(R)
    R_inv = np.linalg.inv(R)
    homo = np.ones(shape=XYZ_landmarks.shape[0])
    XYZ_landmarks = np.concatenate([XYZ_landmarks, np.expand_dims(homo, 1)], axis=1)
    XYZ_wrt_wrist = np.matmul(R_inv, XYZ_landmarks.T)
    XYZ_wrt_wrist = XYZ_wrt_wrist.T
    wrist_XYZ, fingers_XYZ_wrt_wrist = XYZ_wrt_wrist[0, :-1], XYZ_wrt_wrist[1:, :-1]
    fingers_XYZ_wrt_wrist = fingers_XYZ_wrt_wrist.reshape(5, 4, 3)
    return wrist_XYZ, fingers_XYZ_wrt_wrist

def calculate_angles_between_joints(wrist_XYZ: NDArray, fingers_XYZ_wrt_wrist: NDArray, degrees=False) -> NDArray:
    def angle_between(a, b, project_to=None):
        """
        project_to = None means project to "xyz"
        """

        if a.ndim == 1:
            a = np.full_like(b, a)

        mask = [1, 1, 1]
        if project_to is not None:
            assert project_to in ["xy", "xz", "yz"]
            if project_to == "xy":
                #mask = [1, 1, 0]
                a = np.delete(a, -1, axis=1)
                b = np.delete(b, -1, axis=1)
            elif project_to == "xz":
                ##mask = [1, 0, 1]
                a = np.delete(a, 1, axis=1)
                b = np.delete(b, 1, axis=1)
            else:
                #mask = [0, 1, 1]
                a = np.delete(a, 0, axis=1)
                b = np.delete(b, 0, axis=1)
        #a = a * mask
        #b = b * mask

        dot_product = np.sum(a * b, axis=1)  # calculate dot product by element-wise style instead of using np.dot
        magnitude_a = np.linalg.norm(a, axis=1)
        magnitude_b = np.linalg.norm(b, axis=1)
        cos_theta = dot_product / (magnitude_a * magnitude_b)
        angle_radians = np.arccos(cos_theta)
        angle_degrees = np.degrees(angle_radians)

        # Get the direction
        M = np.concatenate([a[:, None, :], b[:, None, :]], axis=1)
        dets = np.linalg.det(M)
        directions = np.sign(dets)
        angle_degrees *= directions

        return angle_degrees

    assert np.sum(np.abs(wrist_XYZ)).astype(np.int8) == 0

    angles = np.zeros(shape=fingers_XYZ_wrt_wrist.shape[:-1])
    vector_y = fingers_XYZ_wrt_wrist[2, 0, :]

    """
    For now, we dont calculate the angles of thumb finger
    """

    # Angles of J11 - > J51
    # The order of params a and b is important here, because we will compute determinant to get the direction
    angles[:, 0] = angle_between(vector_y,
                                    fingers_XYZ_wrt_wrist[:, 1, :] - fingers_XYZ_wrt_wrist[:, 0, :],
                                    project_to="yz")

    # Angles of J12 - > J52
    # The order of params a and b is important here, because we will compute determinant to get the direction
    angles[:, 1] = angle_between(fingers_XYZ_wrt_wrist[:, 0, :],
                                    fingers_XYZ_wrt_wrist[:, 1, :] - fingers_XYZ_wrt_wrist[:, 0, :],
                                    project_to="xy")

    # Angles of J13 - > J53
    # The order of params a and b is important here, because we will compute determinant to get the direction
    angles[:, 2] = angle_between(fingers_XYZ_wrt_wrist[:, 1, :] - fingers_XYZ_wrt_wrist[:, 0, :],
                                    fingers_XYZ_wrt_wrist[:, 2, :] - fingers_XYZ_wrt_wrist[:, 1, :],
                                    project_to="xy")

    # Angles of J14 - > J54
    # The order of params a and b is important here, because we will compute determinant to get the direction
    angles[:, 3] = angle_between(fingers_XYZ_wrt_wrist[:, 2, :] - fingers_XYZ_wrt_wrist[:, 1, :],
                                    fingers_XYZ_wrt_wrist[:, 3, :] - fingers_XYZ_wrt_wrist[:, 2, :],
                                    project_to="xy")

    joint_1_weight = np.interp(np.absolute(angles[:, 1]), [0, 90], [1, 0])
    angles[:, 0] *= joint_1_weight

    #angles = bound_angles(angles, degrees=True)

    if not degrees:
        angles = angles * math.pi / 180                                     
    return angles

def bound_angles(angles, degrees=True):
    joint_1_min_degree, joint_1_max_degree = -35, 35
    joint_2_min_degree, joint_2_max_degree = -100, 8
    joint_3_min_degree, joint_3_max_degree = -100, 8
    joint_4_min_degree, joint_4_max_degree = -100, 8

    if degrees:
        joint_1_min, joint_1_max = joint_1_min_degree, joint_1_max_degree
        joint_2_min, joint_2_max = joint_2_min_degree, joint_2_max_degree
        joint_3_min, joint_3_max = joint_3_min_degree, joint_3_max_degree
        joint_4_min, joint_4_max = joint_4_min_degree, joint_4_max_degree
    else:
        joint_1_min, joint_1_max = joint_1_min_degree * math.pi / 180, joint_1_max_degree * math.pi / 180
        joint_2_min, joint_2_max = joint_2_min_degree * math.pi / 180, joint_2_max_degree * math.pi / 180
        joint_3_min, joint_3_max = joint_3_min_degree * math.pi / 180, joint_3_max_degree * math.pi / 180
        joint_4_min, joint_4_max = joint_4_min_degree * math.pi / 180, joint_4_max_degree * math.pi / 180

    angles[:, 0] = np.clip(angles[:, 0], joint_1_min, joint_1_max)
    angles[:, 1] = np.clip(angles[:, 1], joint_2_min, joint_2_max)
    angles[:, 2] = np.clip(angles[:, 2], joint_3_min, joint_3_max)
    angles[:, 3] = np.clip(angles[:, 3], joint_4_min, joint_4_max)

    return angles

def plot_3d(origin_coords, x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(origin_coords[0], origin_coords[1], origin_coords[2], c='r', marker='*')
    ax.scatter(x.flatten(), y.flatten(), z.flatten(), c='b', marker='o')
    
    # Draw a line
    def draw_line(ax, prev_p, next_p):
        ax.plot([prev_p[0], next_p[0]], [prev_p[1], next_p[1]], [prev_p[2], next_p[2]], c='b')
        return ax

    for finger_i in range(x.shape[0]):
        for joint_j in range(x.shape[1]):
            if joint_j == 0:
                prev_p = origin_coords
            else:
                prev_p = next_p
            next_p = [x[finger_i, joint_j], y[finger_i, joint_j], z[finger_i, joint_j]]
            ax = draw_line(ax, prev_p, next_p)

    # Draw Oxyz coord
    #ax.plot([-20, 200], [0, 0], [0, 0], c='r')
    #ax.plot([0, 0], [0, 200], [0, 0], c='g')
    #ax.plot([0, 0], [0, 0], [-20, 200], c='b')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the limits for each axis
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 100)  # Setting custom limits for z-axis

    fig.savefig('view_1.png')

    ax.view_init(elev=11, azim=-2)
    fig.savefig('view_2.png')
    

    ax.view_init(elev=90, azim=-90)
    fig.savefig('view_3.png')
    
    view_1 = cv2.imread('view_1.png')
    view_2 = cv2.imread('view_2.png')
    view_3 = cv2.imread('view_3.png')
    
    cv2.imshow("View 1", view_1)
    #cv2.moveWindow("View 1", window_sizes[1][0], window_sizes[1][1])
    
    cv2.imshow("View 2", view_2)
    #cv2.moveWindow("View 2", window_sizes[2][0], window_sizes[2][1])
    
    cv2.imshow("View 3", view_3)
    #cv2.moveWindow("View 3", window_sizes[3][0], window_sizes[3][1])

    plt.close()