import yaml
import cv2
import numpy as np
from functools import partial
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import euclidean

from numpy.typing import NDArray
from typing import Tuple

def detect_arm_landmarks(color_img, detector, landmarks_queue, mp_drawing, mp_pose):
    color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

    results = detector.process(color_img)

    print('--------------')
    print(results.pose_landmarks)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(color_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        landmarks_queue.put(results.pose_landmarks)
        if landmarks_queue.qsize() > 1:
            landmarks_queue.get()

    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    return color_img

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

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

def get_xyZ(landmarks, depth, frame_size, sliding_window_size, landmark_ids_to_get=None, visibility_threshold=0.1):
    """
    Input:
        landmark_ids_to_get = None means get all landmarks
    Output:
        xyZ: shape = (N, 3) where N is the num. of landmarks want to get
    """

    assert depth is not None

    if isinstance(landmark_ids_to_get, int):
        landmark_ids_to_get = [landmark_ids_to_get]

    xyz = []

    if landmark_ids_to_get is None:
        for landmark in landmarks.landmark:
            if landmark.visibility > visibility_threshold: 
                x = landmark.x
                y = landmark.y
                z = landmark.z
                xyz.append([x, y, z])
    else:
        for landmark_id in landmark_ids_to_get:
            landmark = landmarks.landmark[landmark_id]
            if landmark.visibility > visibility_threshold: 
                x = landmark.x
                y = landmark.y
                z = landmark.z
                xyz.append([x, y, z])

    xyz = np.array(xyz)
    xy_unnorm = unnormalize(xyz, frame_size)
    Z = get_depth(xy_unnorm, depth, sliding_window_size)
    xyZ = np.concatenate([xy_unnorm, Z[:, None]], axis=-1)
    return xyZ

def get_landmarks_name_based_on_arm(arm_to_get="left"):
    assert arm_to_get in ["left", "right"]

    landmarks_name = ["left shoulder", "left elbow", "left wrist",
                      "left pinky", "left index", "left thumb", "left hip"]
    landmarks_to_visualize = ["right shoulder", "right hip"]

    if arm_to_get == "right":
        landmarks_name = [name.replace("left", "right") for name in landmarks_name]
        landmarks_to_visualize = [name.replace("right", "left") for name in landmarks_to_visualize]

    landmarks_name.extend(landmarks_to_visualize)

    return landmarks_name

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

def xyZ_to_XYZ(xyZ, intrinsic_mat):
    """
    Input:
        xyZ: shape = (N, 3) or (M, N, 3)
    Output:
        XYZ: shape = (N, 3) or (M, N, 3)
    """

    XYZ = np.zeros_like(xyZ)
    XYZ[..., 0] = (xyZ[..., 0] - intrinsic_mat[0, -1]) * xyZ[..., -1] / intrinsic_mat[0, 0]
    XYZ[..., 1] = (xyZ[..., 1] - intrinsic_mat[1, -1]) * xyZ[..., -1] / intrinsic_mat[1, 1]
    XYZ[..., -1] = xyZ[..., -1]

    return XYZ 

def fuse_landmarks_from_two_cameras(opposite_xyZ: NDArray, 
                                    right_side_xyZ: NDArray,
                                    right_side_cam_intrinsic,
                                    opposite_cam_intrinsic,
                                    right_to_opposite_correctmat) -> NDArray:
    """
    Input:
        opposite_xyZ: shape = (N, 3)
        right_side_xyZ: shape = (N, 3)
        right_to_opposite_correctmat: shape = (4, 4)
    Output:
        fused_landmarks: shape (N, 3)
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

def convert_to_shoulder_coord(XYZ_landmarks: NDArray, landmark_dictionary) -> Tuple[NDArray, NDArray]:
    """
    Input: 
        XYZ_landmarks: (N, 3)
    Output:
        wrist_XYZ: (3,)
        fingers_XYZ_wrt_wrist: (a, b, 3) where a * b = (N - 1)
    """

    u = XYZ_landmarks[landmark_dictionary.index("right shoulder"), :] - XYZ_landmarks[landmark_dictionary.index("left shoulder"), :]
    v = XYZ_landmarks[landmark_dictionary.index("left hip"), :] - XYZ_landmarks[landmark_dictionary.index("left shoulder"), :]

    y = np.cross(u, v)
    x = np.cross(u, y)
    z = np.cross(x, y)

    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    z = z / np.linalg.norm(z)

    w_c = XYZ_landmarks[landmark_dictionary.index("left shoulder"), :]

    R = np.array([x, y, z, w_c])
    R = np.concatenate([R, np.expand_dims([0, 0, 0, 1], 1)], axis=1)
    R = np.transpose(R)
    R_inv = np.linalg.inv(R)
    homo = np.ones(shape=XYZ_landmarks.shape[0])
    XYZ_landmarks = np.concatenate([XYZ_landmarks, np.expand_dims(homo, 1)], axis=1)
    XYZ_wrt_shoulder = np.matmul(R_inv, XYZ_landmarks.T)
    XYZ_wrt_shoulder = XYZ_wrt_shoulder.T
    XYZ_wrt_shoulder = XYZ_wrt_shoulder[..., :-1]
    #shoulder_XYZ, arm_and_body_XYZ_wrt_shoulder = XYZ_wrt_shoulder[0, :-1], XYZ_wrt_shoulder[1:, :-1]
    #arm_and_body_XYZ_wrt_shoulder = arm_and_body_XYZ_wrt_shoulder.reshape(5, 4, 3)
    return XYZ_wrt_shoulder 