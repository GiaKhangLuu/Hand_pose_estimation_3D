import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.typing import NDArray
from typing import Tuple
from functools import partial
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import euclidean
from scipy.optimize import minimize
from typing import Tuple

finger_joints_names = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]

def load_data_from_npz_file(file_path):
    data = np.load(file_path)

    return data

def get_oak_2_rs_matrix(oak_r_raw, oak_t_raw, 
                        rs_r_raw, rs_t_raw):
    rs_r_raw = rs_r_raw.squeeze()
    rs_t_raw = rs_t_raw.squeeze()
    oak_r_raw = oak_r_raw.squeeze()
    oak_t_raw = oak_t_raw.squeeze()

    rs_r_mat = R.from_rotvec(rs_r_raw, degrees=False)
    rs_r_mat = rs_r_mat.as_matrix()

    oak_r_mat = R.from_rotvec(oak_r_raw, degrees=False)
    oak_r_mat = oak_r_mat.as_matrix()

    oak_r_t_mat = np.dstack([oak_r_mat, oak_t_raw[:, :, None]])
    rs_r_t_mat = np.dstack([rs_r_mat, rs_t_raw[:, :, None]])

    extra_row = np.array([0, 0, 0, 1] * oak_r_t_mat.shape[0]).reshape(-1, 4)[:, None, :]
    oak_r_t_mat = np.concatenate([oak_r_t_mat, extra_row], axis=1)
    rs_r_t_mat = np.concatenate([rs_r_t_mat, extra_row], axis=1)

    oak_r_t_mat_inv = np.linalg.inv(oak_r_t_mat)
    oak_2_rs_mat = np.matmul(rs_r_t_mat, oak_r_t_mat_inv)
    oak_2_rs_mat_avg = np.average(oak_2_rs_mat, axis=0)

    return oak_2_rs_mat_avg

def get_xyZ(landmarks, depth, frame_size, sliding_window_size):
    """
    Output:
        xyZ: shape = (21, 3)
    """

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
    """
    Input:
        right_side_xyZ: shape = (3,)
        opposite_xyZ: shape = (3,)
        right_to_opposite_correctmat: shape = (4, 4)
    """

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
    """
    Input:
        opposite_xyZ: shape = (21, 3)
        right_side_xyZ: shape = (21, 3)
        right_to_opposite_correctmat: shape = (4, 4)
    Output:
        fused_landmarks: shape (21, 3)
    """

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

    right_side_new_XYZ = xyZ_to_XYZ(right_side_new_xyZ, right_side_cam_intrinsic)
    opposite_new_XYZ = xyZ_to_XYZ(opposite_new_xyZ, opposite_cam_intrinsic)

    fused_landmarks = (right_side_new_XYZ + opposite_new_XYZ) / 2

    return fused_landmarks

def xyZ_to_XYZ(xyZ, intrinsic_mat):
    """
    Input:
        xyZ: shape = (21, 3)
    Output:
        XYZ: shape = (21, 3)
    """

    XYZ = np.zeros_like(xyZ)
    XYZ[:, 0] = (xyZ[:, 0] - intrinsic_mat[0, -1]) * xyZ[:, -1] / intrinsic_mat[0, 0]
    XYZ[:, 1] = (xyZ[:, 1] - intrinsic_mat[1, -1]) * xyZ[:, -1] / intrinsic_mat[1, 1]
    XYZ[:, -1] = xyZ[:, -1]

    return XYZ 

def convert_to_wrist_coord(XYZ_landmarks: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Input: 
        XYZ_landmarks: (21, 3)
    Output:
        wrist_XYZ: (3,)
        fingers_XYZ_wrt_wrist: (5, 4, 3)
    """

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
