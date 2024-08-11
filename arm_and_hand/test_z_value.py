import sys
import os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, '..'))

import yaml
import cv2
import mediapipe as mp
import numpy as np
import queue
import threading
import time
from functools import partial
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from filterpy.kalman import KalmanFilter

from stream_3d import visualize_arm, visualize_hand
from angle_calculation import get_angles_between_joints
from mediapipe_drawing import draw_hand_landmarks_on_image, draw_arm_landmarks_on_image
from camera_tools import initialize_oak_cam, initialize_realsense_cam, stream_rs, stream_oak
from utilities import (filter_depth,
    detect_hand_landmarks,
    detect_arm_landmarks,
    get_normalized_and_world_pose_landmarks,
    get_normalized_and_world_hand_landmarks,
    get_landmarks_name_based_on_arm,
    fuse_landmarks_from_two_cameras,
    convert_to_shoulder_coord,
    convert_to_wrist_coord,
    get_mediapipe_world_landmarks)
from common import (load_data_from_npz_file,
    get_oak_2_rs_matrix,
    load_config,
    get_xyZ)

# ------------------- READ CONFIG ------------------- 
config_file_path = os.path.join(CURRENT_DIR, "configuration.yaml") 
config = load_config(config_file_path)

left_oak_mxid = config["camera"]["left_camera"]["mxid"]
left_oak_stereo_size = (config["camera"]["left_camera"]["stereo"]["width"], 
    config["camera"]["left_camera"]["stereo"]["height"]) 
left_cam_calib_path = config["camera"]["left_camera"]["left_camera_calibration_path"]

right_oak_mxid = config["camera"]["right_camera"]["mxid"]
right_oak_stereo_size = (config["camera"]["right_camera"]["stereo"]["width"], 
    config["camera"]["right_camera"]["stereo"]["height"]) 
right_cam_calib_path = config["camera"]["right_camera"]["right_camera_calibration_path"]

frame_size = (config["process_frame"]["frame_size"]["width"], 
    config["process_frame"]["frame_size"]["height"])
frame_color_format = config["process_frame"]["frame_color_format"]

plot_3d = config["utilities"]["plot_3d"]
is_draw_landmarks = config["utilities"]["draw_landmarks"]
save_landmarks = config["utilities"]["save_landmarks"]

arm_detection_activation = config["arm_landmark_detection"]["is_enable"]
arm_min_det_conf = config["arm_landmark_detection"]["min_detection_confidence"]
arm_min_tracking_conf = config["arm_landmark_detection"]["min_tracking_confidence"]
arm_visibility_threshold = config["arm_landmark_detection"]["visibility_threshold"]
arm_num_pose_detected = config["arm_landmark_detection"]["num_pose"]
arm_model_path = config["arm_landmark_detection"]["model_asset_path"]
arm_pose_landmarks_name = tuple(config["arm_landmark_detection"]["pose_landmarks"])
arm_fuse_left = config["arm_landmark_detection"]["arm_to_fuse"]["left"]
arm_fuse_right = config["arm_landmark_detection"]["arm_to_fuse"]["right"]
arm_fuse_both = config["arm_landmark_detection"]["arm_to_fuse"]["both"]

hand_detection_activation = config["hand_landmark_detection"]["is_enable"]
hand_min_det_conf = config["hand_landmark_detection"]["min_detection_confidence"]
hand_min_tracking_conf = config["hand_landmark_detection"]["min_tracking_confidence"]
hand_min_presence_conf = config["hand_landmark_detection"]["min_presence_confidence"]
hand_num_hand_detected = config["hand_landmark_detection"]["num_hand"]
hand_model_path = config["hand_landmark_detection"]["model_asset_path"]
hand_fuse_left = config["hand_landmark_detection"]["hand_to_fuse"]["left"]
hand_fuse_right = config["hand_landmark_detection"]["hand_to_fuse"]["right"]
hand_fuse_both = config["hand_landmark_detection"]["hand_to_fuse"]["both"]
hand_landmarks_name = tuple(config["hand_landmark_detection"]["hand_landmarks"])

draw_xyz = config["debugging_mode"]["draw_xyz"]
debug_angle_j1 = config["debugging_mode"]["show_left_arm_angle_j1"]
debug_angle_j2 = config["debugging_mode"]["show_left_arm_angle_j2"]
debug_angle_j3 = config["debugging_mode"]["show_left_arm_angle_j3"]
debug_angle_j4 = config["debugging_mode"]["show_left_arm_angle_j4"]
debug_angle_j5 = config["debugging_mode"]["show_left_arm_angle_j5"]
debug_angle_j6 = config["debugging_mode"]["show_left_arm_angle_j6"]
ref_vector_color = list(config["debugging_mode"]["ref_vector_color"])
joint_vector_color = list(config["debugging_mode"]["joint_vector_color"])

# -------------------- GET TRANSFORMATION MATRIX -------------------- 
right_cam_data = load_data_from_npz_file(right_cam_calib_path)

right_oak_r_raw, right_oak_t_raw, right_oak_ins = right_cam_data['rvecs'], right_cam_data['tvecs'], right_cam_data['camMatrix']

def _scale_intrinsic_by_res(intrinsic, original_size, processed_size):
    """
    Default camera's resolution is different with processing resolution. Therefore,
    we need another intrinsic that is compatible with the processing resolution.
    """
    original_h, original_w = original_size
    processed_h, processed_w = processed_size
    scale_w = processed_w / original_w
    scale_h = processed_h / original_h
    intrinsic[0, :] = intrinsic[0, :] * scale_w
    intrinsic[1, :] = intrinsic[1, :] * scale_h

    return intrinsic

original_size = (1080, 1920)
right_oak_ins = _scale_intrinsic_by_res(right_oak_ins, original_size, frame_size[::-1])
#left_oak_ins = _scale_intrinsic_by_res(left_oak_ins, original_size, frame_size[::-1])

#right_2_left_mat_avg = get_oak_2_rs_matrix(right_oak_r_raw, right_oak_t_raw, 
    #left_oak_r_raw, left_oak_t_raw)

# -------------------- INIT MEDIAPIPE MODELS -------------------- 
hand_base_options = python.BaseOptions(model_asset_path=hand_model_path,
    delegate=mp.tasks.BaseOptions.Delegate.GPU)
hand_options = vision.HandLandmarkerOptions(
    base_options=hand_base_options,
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=hand_num_hand_detected,
    min_hand_detection_confidence=hand_min_det_conf,
    min_hand_presence_confidence=hand_min_presence_conf,
    min_tracking_confidence=hand_min_tracking_conf)
oak_hand_detector = vision.HandLandmarker.create_from_options(hand_options)

arm_base_options = python.BaseOptions(model_asset_path=arm_model_path,
    delegate=mp.tasks.BaseOptions.Delegate.GPU)
arm_options = vision.PoseLandmarkerOptions(
    base_options=arm_base_options,
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_poses=arm_num_pose_detected,
    min_pose_detection_confidence=arm_min_det_conf,
    min_tracking_confidence=arm_min_tracking_conf,
    output_segmentation_masks=False)
oak_arm_detector = vision.PoseLandmarker.create_from_options(arm_options)  # Each camera is responsible for one detector because the detector stores the pose in prev. frame

# -------------------- CREATE QUEUE TO INTERACT WITH THREADS -------------------- 
rightside_frame_getter_queue = queue.Queue()  # This queue directly gets RAW frame (rgb, depth) from OAK camera

arm_points_vis_queue = queue.Queue()  # This queue stores fused arm landmarks to visualize by open3d
hand_points_vis_queue = queue.Queue()  # This queue stores fused hand landmarks to visualize by open3d 

hand_to_fuse = "Left" if hand_fuse_left else "Right"
arm_to_get = "left" if arm_fuse_left else "right"
landmarks_name_want_to_get = get_landmarks_name_based_on_arm(arm_to_get)
landmarks_name_want_to_get = ["left shoulder", "right shoulder", "left elbow", "right elbow", "left wrist", "right wrist"]
landmarks_id_want_to_get = [arm_pose_landmarks_name.index(name) for name in landmarks_name_want_to_get]

arm_hand_fused_names = landmarks_name_want_to_get.copy()
arm_hand_fused_names.extend(hand_landmarks_name) 

def get_mediapipe_landmarks(landmarks, landmark_ids_to_get=None, visibility_threshold=None):
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
    return XYZ

if __name__ == "__main__":
    pipeline_right_oak = initialize_oak_cam(right_oak_stereo_size)

    right_oak_thread = threading.Thread(target=stream_oak, args=(pipeline_right_oak,  
        rightside_frame_getter_queue, right_oak_mxid), daemon=True)

    vis_arm_thread = threading.Thread(target=visualize_arm,
        args=(arm_points_vis_queue, 
            arm_hand_fused_names,
            debug_angle_j1,
            debug_angle_j2,
            debug_angle_j3,
            debug_angle_j4,
            debug_angle_j5,
            debug_angle_j6,
            draw_xyz,
            True,
            joint_vector_color,
            ref_vector_color,),
        daemon=True)

    vis_hand_thread = threading.Thread(target=visualize_hand,
        args=(hand_points_vis_queue,),
        daemon=True)

    right_oak_thread.start()
    if plot_3d:
        vis_arm_thread.start()

    timestamp = 0
    frame_count = 0
    start_time = time.time()
    fps = 0

    while True:
        rs_hand_landmarks, rs_handedness = None, None
        oak_hand_landmarks, oak_handedness = None, None
        rs_arm_landmarks = None
        oak_arm_landmarks = None
        arm_fused_XYZ, hand_fused_XYZ = None, None 

        # 1. Get and preprocess rgb and its depth
        rightside_rgb, rightside_depth = rightside_frame_getter_queue.get()

        rightside_rgb = cv2.resize(rightside_rgb, frame_size)
        rightside_depth = cv2.resize(rightside_depth, frame_size)

        assert rightside_depth.shape == rightside_rgb.shape[:-1]

        # SYNCHRONOUSLY PERFORM
        processed_oak_img = np.copy(rightside_rgb)
        processed_oak_img = cv2.convertScaleAbs(processed_oak_img, alpha=1.0, beta=-50)
        if frame_color_format == "bgr":
            processed_oak_img = cv2.cvtColor(processed_oak_img, cv2.COLOR_BGR2RGB)

        mp_oak_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=processed_oak_img)

        oak_arm_result = oak_arm_detector.detect_for_video(mp_oak_image, timestamp)
        oak_arm_landmarks, oak_arm_world_landmarks = get_normalized_and_world_pose_landmarks(oak_arm_result)

        rightside_rgb = draw_arm_landmarks_on_image(rightside_rgb, oak_arm_landmarks)

        right_cam_arm_landmarks_XYZ = None
        if oak_arm_landmarks and oak_arm_world_landmarks:
            if arm_num_pose_detected == 1:
                oak_arm_landmarks = oak_arm_landmarks[0]
            right_cam_arm_landmarks_XYZ = get_mediapipe_landmarks(oak_arm_landmarks,  # shape: (N, 3)
                landmark_ids_to_get=landmarks_id_want_to_get,
                visibility_threshold=arm_visibility_threshold)  

        frame_count += 1
        elapsed_time = time.time() - start_time

        # Update FPS every second
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        rightside_rgb = cv2.resize(rightside_rgb, (960, 540))

        if right_cam_arm_landmarks_XYZ is not None:
            for i in range(right_cam_arm_landmarks_XYZ.shape[0]):
                x, y, z = right_cam_arm_landmarks_XYZ[i, 0], right_cam_arm_landmarks_XYZ[i, 1], right_cam_arm_landmarks_XYZ[i, 2]
                x = int(x * 960)
                y = int(y * 540)
                z = int(z * 960)

                cv2.putText(rightside_rgb, f'z: {z}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)


        cv2.putText(rightside_rgb, f'FPS: {fps:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4, cv2.LINE_AA)

        cv2.imshow("Frame", rightside_rgb)

        timestamp += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break