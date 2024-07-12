import yaml
import cv2
import numpy as np
import mediapipe as mp
import time
from functools import partial
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import euclidean

from numpy.typing import NDArray
from typing import Tuple, List

from mediapipe.framework.formats import landmark_pb2
import numpy as np

def filter_depth(depth_array: NDArray, sliding_window_size, sigma_color, sigma_space) -> NDArray:
    depth_array = depth_array.astype(np.float32)
    depth_array = cv2.bilateralFilter(depth_array, sliding_window_size, sigma_color, sigma_space)
    return depth_array

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

def get_normalized_pose_landmarks(detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    pose_landmarks_proto_list = []
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

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
    handedness_list = detection_result.handedness
    hand_landmarks_proto_list = []
    hand_info_list = []

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
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

def get_landmarks_name_based_on_arm(arm_to_get="left"):
	"""
	Currently support for left OR right arm
	"""

	assert arm_to_get in ["left", "right"]

	landmarks_name = ["left shoulder", "left elbow", "left wrist",
        "left pinky", "left index", "left thumb", "left hip"]
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

        min_dis = partial(distance, 
			right_side_xyZ=right_side_i_xyZ, 
			opposite_xyZ=opposite_i_xyZ,
            right_side_cam_intrinsic=right_side_cam_intrinsic,
            opposite_cam_intrinsic=opposite_cam_intrinsic,
            right_to_opposite_correctmat=right_to_opposite_correctmat)
        result = minimize(min_dis, 
            x0=[right_side_i_xyZ[-1], opposite_i_xyZ[-1]], 
            tol=1e-1,
            method="SLSQP")
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
        XYZ_wrt_shoulder: (N, 3)
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

    R = np.array([x_unit, y_unit, z_unit, w_c])
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