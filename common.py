import numpy as np
import yaml
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import euclidean

from numpy.typing import NDArray
from typing import Tuple, List

def load_data_from_npz_file(file_path):
    data = np.load(file_path)
    return data

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

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

def unnormalize(arr: NDArray, frame_size) -> NDArray:
    #arr = arr[:, :-1]
    arr[:, 0] = arr[:, 0] * frame_size[0]
    arr[:, 1] = arr[:, 1] * frame_size[1]
    arr[:, 2] = arr[:, 2] * frame_size[0]

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
        if np.isnan(z_median) or np.isnan(z_median):
            z_median = 0
        z_landmarks.append(z_median)

    return np.array(z_landmarks)

def get_xyZ(landmarks, frame_size, landmark_ids_to_get=None, visibility_threshold=None, depth_map=None):
    """
    Input:
        landmark_ids_to_get = None means get all landmarks
        visibility_threshold = None means we dont consider checking visibility
    Output:
        xyZ: shape = (N, 3) where N is the num. of landmarks want to get
    """

    if isinstance(landmark_ids_to_get, int):
        landmark_ids_to_get = [landmark_ids_to_get]

    xyz = []
    if landmark_ids_to_get is None:
        for landmark in landmarks.landmark:
            if (visibility_threshold is None or 
                landmark.visibility > visibility_threshold):
                x = landmark.x
                y = landmark.y
                z = landmark.z
                xyz.append([x, y, z])
    else:
        for landmark_id in landmark_ids_to_get:
            landmark = landmarks.landmark[landmark_id]
            if (visibility_threshold is None or
                landmark.visibility > visibility_threshold):
                x = landmark.x
                y = landmark.y
                z = landmark.z
                xyz.append([x, y, z])

    if not len(xyz):
        return None

    xyz = np.array(xyz)
    xyz_unnorm = unnormalize(xyz, frame_size)

    if depth_map is not None:
        xy_unnorm = xyz_unnorm[:, :-1]
        Z = get_depth(xy_unnorm, depth_map, 9)
        xyZ = np.concatenate([xy_unnorm, Z[:, None]], axis=-1)
        return xyZ
    return xyz_unnorm

def scale_intrinsic_by_res(intrinsic, calibrated_size, processed_size):
    """
    Default camera's resolution is different with processing resolution. Therefore,
    we need another intrinsic that is compatible with the processing resolution.
    """
    assert intrinsic.ndim in [2, 3]
    if intrinsic.ndim == 2:
        calibrated_h, calibrated_w = calibrated_size
        processed_h, processed_w = processed_size
        scale_w = processed_w / calibrated_w
        scale_h = processed_h / calibrated_h
        intrinsic[0, :] = intrinsic[0, :] * scale_w
        intrinsic[1, :] = intrinsic[1, :] * scale_h
    else:
        assert calibrated_size.ndim == processed_size.ndim == 2
        calibrated_h, calibrated_w = calibrated_size[..., 0], calibrated_size[..., 1]
        processed_h, processed_w = processed_size[..., 0], processed_size[..., 1]
        scale_w = processed_w / calibrated_w
        scale_h = processed_h / calibrated_h
        intrinsic[:, 0, :] *= scale_w[:, None]
        intrinsic[:, 1, :] *= scale_h[:, None]

    return intrinsic