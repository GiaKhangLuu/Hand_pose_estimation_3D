import yaml
import cv2
import numpy as np
import mediapipe as mp
import time
from functools import partial
from scipy.optimize import minimize, Bounds
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import euclidean

from numpy.typing import NDArray
from typing import Tuple, List

from mediapipe.framework.formats import landmark_pb2
import numpy as np

def detect_arm_landmarks(rs_detector, oak_detector, input_queue, result_queue, image_format="bgr"):
    while True:
        if not input_queue.empty():
            rs_color_img, oak_color_img, timestamp = input_queue.get()

            processed_rs_img = np.copy(rs_color_img) 
            processed_oak_img = np.copy(oak_color_img)
            if image_format == "bgr":
                processed_rs_img = cv2.cvtColor(processed_rs_img, cv2.COLOR_BGR2RGB)
                processed_oak_img = cv2.cvtColor(processed_oak_img, cv2.COLOR_BGR2RGB)

            mp_rs_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=processed_rs_img)
            mp_oak_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=processed_oak_img)

            rs_result = rs_detector.detect_for_video(mp_rs_image, timestamp)
            oak_result = oak_detector.detect_for_video(mp_oak_image, timestamp)
            #rs_result = rs_detector.detect(mp_rs_image)
            #oak_result = oak_detector.detect(mp_oak_image)

            if rs_result.pose_landmarks and oak_result.pose_landmarks:
                result_queue.put((rs_result, oak_result))
                if result_queue.qsize() > 1:
                    result_queue.get()

        time.sleep(0.0001)

def detect_hand_landmarks(rs_detector, oak_detector, input_queue, result_queue, image_format="bgr"):
    while True:
        if not input_queue.empty():
            rs_color_img, oak_color_img, timestamp = input_queue.get()

            processed_rs_img = np.copy(rs_color_img) 
            processed_oak_img = np.copy(oak_color_img)
            if image_format == "bgr":
                processed_rs_img = cv2.cvtColor(processed_rs_img, cv2.COLOR_BGR2RGB)
                processed_oak_img = cv2.cvtColor(processed_oak_img, cv2.COLOR_BGR2RGB)

            mp_rs_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=processed_rs_img)
            mp_oak_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=processed_oak_img)

            rs_result = rs_detector.detect_for_video(mp_rs_image, timestamp)
            oak_result = oak_detector.detect_for_video(mp_oak_image, timestamp)

            if rs_result.hand_landmarks and oak_result.hand_landmarks:
                result_queue.put((rs_result, oak_result))
                if result_queue.qsize() > 1:
                    result_queue.get()

        time.sleep(0.0001)

def get_normalized_and_world_pose_landmarks(detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    pose_world_landmarks_list = detection_result.pose_world_landmarks
    pose_landmarks_proto_list = []
    pose_world_landmarks_proto_list = []
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_world_landmarks_proto = landmark_pb2.LandmarkList()

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        pose_world_landmarks = pose_world_landmarks_list[idx]

        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, 
                                            y=landmark.y, 
                                            z=landmark.z,
                                            visibility=landmark.visibility) 
                                            for landmark in pose_landmarks
        ])

        pose_world_landmarks_proto.landmark.extend([
            landmark_pb2.Landmark(x=landmark.x,
                y=landmark.y,
                z=landmark.z,
                visibility=landmark.visibility) for landmark in pose_world_landmarks
        ])

        pose_world_landmarks_proto_list.append(pose_world_landmarks_proto)
        pose_landmarks_proto_list.append(pose_landmarks_proto)

    return pose_landmarks_proto_list, pose_world_landmarks_proto_list

def get_normalized_pose_landmarks(detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    pose_world_landmarks_list = detection_result.pose_world_landmarks
    pose_landmarks_proto_list = []
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        pose_world_landmarks = pose_world_landmarks_list[idx]

        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, 
                                            y=landmark.y, 
                                            z=landmark.z,
                                            visibility=landmark.visibility) 
                                            for landmark in pose_landmarks
        ])

        pose_landmarks_proto_list.append(pose_landmarks_proto)

    return pose_landmarks_proto_list 

def get_normalized_hand_landmarks(detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    hand_world_landmarks_list = detection_result.hand_world_landmarks
    handedness_list = detection_result.handedness
    hand_landmarks_proto_list = []
    hand_info_list = []

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        hand_world_landmarks = hand_world_landmarks_list[idx]
        handedness = handedness_list[idx]

        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
			landmark_pb2.NormalizedLandmark(
				x=landmark.x, 
				y=landmark.y, 
				z=landmark.z) for landmark in hand_landmarks])
        
        hand_landmarks_proto_list.append(hand_landmarks_proto)
        hand_info_list.append(handedness[0].category_name)

    return hand_landmarks_proto_list, hand_info_list

def get_normalized_and_world_hand_landmarks(detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    hand_world_landmarks_list = detection_result.hand_world_landmarks
    handedness_list = detection_result.handedness
    hand_landmarks_proto_list = []
    hand_world_landmarks_proto_list = []
    hand_info_list = []

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        hand_world_landmarks = hand_world_landmarks_list[idx]
        handedness = handedness_list[idx]

        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
			landmark_pb2.NormalizedLandmark(
				x=landmark.x, 
				y=landmark.y, 
				z=landmark.z) for landmark in hand_landmarks])
        
        hand_world_landmarks_proto = landmark_pb2.LandmarkList()
        hand_world_landmarks_proto.landmark.extend([
            landmark_pb2.Landmark(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z) for landmark in hand_world_landmarks])

        hand_landmarks_proto_list.append(hand_landmarks_proto)
        hand_world_landmarks_proto_list.append(hand_world_landmarks_proto)
        hand_info_list.append(handedness[0].category_name)

    return hand_landmarks_proto_list, hand_world_landmarks_proto_list, hand_info_list

def get_landmarks_name_based_on_arm(arm_to_get="left"):
	"""
	Currently support for left OR right arm
	"""

	assert arm_to_get in ["left", "right"]

	landmarks_name = ["left shoulder", "left elbow", "left hip"]
	landmarks_to_visualize = ["right shoulder", "right hip"]

	if arm_to_get == "right":
		landmarks_name = [name.replace("left", "right") for name in landmarks_name]
		landmarks_to_visualize = [name.replace("right", "left") for name in landmarks_to_visualize]

	landmarks_name.extend(landmarks_to_visualize)

	return landmarks_name

def distance(Z, 
	right_side_xyZ, 
	opposite_xyZ, 
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
    right_to_opposite_correctmat,
    tolerance,
    method_name) -> NDArray:
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

        min_dis = partial(distance, 
			right_side_xyZ=right_side_i_xyZ, 
			opposite_xyZ=opposite_i_xyZ,
            right_side_cam_intrinsic=right_side_cam_intrinsic,
            opposite_cam_intrinsic=opposite_cam_intrinsic,
            right_to_opposite_correctmat=right_to_opposite_correctmat)
        result = minimize(min_dis, 
            tol=tolerance, 
            x0=[right_side_i_xyZ[-1], opposite_i_xyZ[-1]],
            method=method_name)
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
        XYZ_landmarks: (M, 3), M = #vectors, 3 = #features
    Output:
        XYZ_wrt_shoulder: (M, 3), M = #vectors, 3 = #features
        origin_xyz_wrt_shoulder: (3, O), 3 = #features, O = #vectors (xyz)
    """

    u = XYZ_landmarks[landmark_dictionary.index("right shoulder"), :] - XYZ_landmarks[landmark_dictionary.index("left shoulder"), :]
    v = XYZ_landmarks[landmark_dictionary.index("left hip"), :] - XYZ_landmarks[landmark_dictionary.index("left shoulder"), :]

    y = np.cross(u, v)
    x = np.cross(u, y)
    z = np.cross(x, y)

    x_unit = x / np.linalg.norm(x)
    y_unit = y / np.linalg.norm(y)
    z_unit = z / np.linalg.norm(z)

    w_c = XYZ_landmarks[landmark_dictionary.index("left shoulder"), :]

    R = np.array([x_unit, y_unit, z_unit, w_c])  # (4, 3), 4 = #vectors, 3 = #features
    R = np.concatenate([R, np.expand_dims([0, 0, 0, 1], 1)], axis=1)  # (4, 4)
    R = np.transpose(R)  # (4, 4) 
    R_inv = np.linalg.inv(R)  # (4, 4)
    homo = np.ones(shape=XYZ_landmarks.shape[0])
    XYZ_landmarks = np.concatenate([XYZ_landmarks, np.expand_dims(homo, 1)], axis=1)  # (N, 4)
    XYZ_wrt_shoulder = np.matmul(R_inv, XYZ_landmarks.T)  # (4, N), 4 = #features, N = #vectors
    XYZ_wrt_shoulder = XYZ_wrt_shoulder.T  # (N, 4), 4 = #features, N = #vectors
    XYZ_wrt_shoulder = XYZ_wrt_shoulder[..., :-1]  # (N, 3), 3 = #features, N = #vectors

    origin_xyz_wrt_shoulder = np.matmul(R_inv, R)  # (4, 4)
    #origin_xyz_wrt_shoulder = origin_xyz_wrt_shoulder.T  # (4, 4)
    origin_xyz_wrt_shoulder = origin_xyz_wrt_shoulder[:-1, :-1]  # (3, 3), 3 = #features, 3 = #vectors

    return XYZ_wrt_shoulder, origin_xyz_wrt_shoulder

def convert_to_wrist_coord(XYZ_landmarks: NDArray, finger_joints_names) -> Tuple[NDArray, NDArray]:
    """
    Input: 
        XYZ_landmarks: (21, 3)
    Output:
        XYZ_wrt_wrist: (21, 3)
    """

    u = XYZ_landmarks[finger_joints_names.index("INDEX_FINGER_MCP"), :] - XYZ_landmarks[finger_joints_names.index("WRIST"), :]
    y = XYZ_landmarks[finger_joints_names.index("MIDDLE_FINGER_MCP"), :] - XYZ_landmarks[finger_joints_names.index("WRIST"), :]

    x = np.cross(y, u)
    z = np.cross(x, y)

    x_unit = x / np.linalg.norm(x)
    y_unit = y / np.linalg.norm(y)
    z_unit = z / np.linalg.norm(z)

    w_c = XYZ_landmarks[finger_joints_names.index("WRIST"), :]

    R = np.array([x_unit, y_unit, z_unit, w_c])
    R = np.concatenate([R, np.expand_dims([0, 0, 0, 1], 1)], axis=1)
    R = np.transpose(R)
    R_inv = np.linalg.inv(R)
    homo = np.ones(shape=XYZ_landmarks.shape[0])
    XYZ_landmarks = np.concatenate([XYZ_landmarks, np.expand_dims(homo, 1)], axis=1)
    XYZ_wrt_wrist = np.matmul(R_inv, XYZ_landmarks.T)
    XYZ_wrt_wrist = XYZ_wrt_wrist.T
    XYZ_wrt_wrist = XYZ_wrt_wrist[..., :-1]
    #wrist_XYZ, fingers_XYZ_wrt_wrist = XYZ_wrt_wrist[0, :-1], XYZ_wrt_wrist[1:, :-1]
    #fingers_XYZ_wrt_wrist = fingers_XYZ_wrt_wrist.reshape(5, 4, 3)
    return XYZ_wrt_wrist

def get_mediapipe_world_landmarks(landmarks, meters_to_millimeters=True, landmark_ids_to_get=None, visibility_threshold=None):
    """
    Input:
        landmark_ids_to_get = None means get all landmarks
        visibility_threshold = None means we dont consider checking visibility
    Output:
        xyz: shape = (N, 3) where N is the num. of landmarks want to get
    """

    if isinstance(landmark_ids_to_get, int):
        landmark_ids_to_get = [landmark_ids_to_get]

    XYZ = []
    if landmark_ids_to_get is None:
        for landmark in landmarks.landmark:
            if (visibility_threshold is None or 
                landmark.visibility > visibility_threshold):
                x = landmark.x
                y = landmark.y
                z = landmark.z
                XYZ.append([x, y, z])
    else:
        for landmark_id in landmark_ids_to_get:
            landmark = landmarks.landmark[landmark_id]
            if (visibility_threshold is None or
                landmark.visibility > visibility_threshold):
                x = landmark.x
                y = landmark.y
                z = landmark.z
                XYZ.append([x, y, z])
    if not len(XYZ):
        return None
    XYZ = np.array(XYZ)
    if meters_to_millimeters:
        XYZ = XYZ * 1000
    return XYZ

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
        z_median = 0 if np.isnan(z_median) else z_median
        z_landmarks.append(z_median)

    return np.array(z_landmarks)

def get_xyZ(landmarks, depth, frame_size, sliding_window_size, landmark_ids_to_get=None, visibility_threshold=None):
    """
    Input:
        landmark_ids_to_get = None means get all landmarks
        visibility_threshold = None means we dont consider checking visibility
    Output:
        xyZ: shape = (N, 3) where N is the num. of landmarks want to get
    """

    assert depth is not None

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
    xy_unnorm = unnormalize(xyz, frame_size)
    Z = get_depth(xy_unnorm, depth, sliding_window_size)
    xyZ = np.concatenate([xy_unnorm, Z[:, None]], axis=-1)
    return xyZ