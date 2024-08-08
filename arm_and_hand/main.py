"""
Convention:
    rs = opposite = REALSENSE camera
    oak = rightside = OAK camera
"""

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
from utilities import (detect_hand_landmarks,
    detect_arm_landmarks,
    get_normalized_and_world_pose_landmarks,
    get_normalized_and_world_hand_landmarks,
    get_normalized_hand_landmarks,
    get_normalized_pose_landmarks,
    get_landmarks_name_based_on_arm,
    fuse_landmarks_from_two_cameras,
    convert_to_shoulder_coord,
    convert_to_wrist_coord,
    get_mediapipe_world_landmarks,
    xyZ_to_XYZ)
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
left_cam_data = load_data_from_npz_file(left_cam_calib_path)

right_oak_r_raw, right_oak_t_raw, right_oak_ins = right_cam_data['rvecs'], right_cam_data['tvecs'], right_cam_data['camMatrix']
left_oak_r_raw, left_oak_t_raw, left_oak_ins = left_cam_data['rvecs'], left_cam_data['tvecs'], left_cam_data['camMatrix']

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

original_size = (720, 1280)
right_oak_ins = _scale_intrinsic_by_res(right_oak_ins, original_size, frame_size[::-1])
left_oak_ins = _scale_intrinsic_by_res(left_oak_ins, original_size, frame_size[::-1])

right_2_left_mat_avg = get_oak_2_rs_matrix(right_oak_r_raw, right_oak_t_raw, 
    left_oak_r_raw, left_oak_t_raw)

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
rs_hand_detector = vision.HandLandmarker.create_from_options(hand_options)
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
rs_arm_detector = vision.PoseLandmarker.create_from_options(arm_options)  # Each camera is responsible for one detector because the detector stores the pose in prev. frame
oak_arm_detector = vision.PoseLandmarker.create_from_options(arm_options)  # Each camera is responsible for one detector because the detector stores the pose in prev. frame

# -------------------- CREATE QUEUE TO INTERACT WITH THREADS -------------------- 
opposite_frame_getter_queue = queue.Queue()  # This queue directly gets RAW frame (rgb, depth) from REALSENSE camera
rightside_frame_getter_queue = queue.Queue()  # This queue directly gets RAW frame (rgb, depth) from OAK camera

arm_points_vis_queue = queue.Queue()  # This queue stores fused arm landmarks to visualize by open3d
hand_points_vis_queue = queue.Queue()  # This queue stores fused hand landmarks to visualize by open3d 

hand_to_fuse = "Left" if hand_fuse_left else "Right"
arm_to_get = "left" if arm_fuse_left else "right"
landmarks_name_want_to_get = get_landmarks_name_based_on_arm(arm_to_get)
landmarks_id_want_to_get = [arm_pose_landmarks_name.index(name) for name in landmarks_name_want_to_get]

arm_hand_fused_names = landmarks_name_want_to_get.copy()
arm_hand_fused_names.extend(hand_landmarks_name) 

if __name__ == "__main__":
    pipeline_left_oak = initialize_oak_cam(left_oak_stereo_size)
    pipeline_right_oak = initialize_oak_cam(right_oak_stereo_size)

    left_oak_thread = threading.Thread(target=stream_oak, args=(pipeline_left_oak,  
        opposite_frame_getter_queue, left_oak_mxid), daemon=True)
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

    left_oak_thread.start()
    right_oak_thread.start()
    if plot_3d:
        vis_arm_thread.start()
        #vis_hand_thread.start()

    timestamp = 0
    frame_count = 0
    start_time = time.time()
    fps = 0

    #kalman_filters = []
    #for i in range(len(arm_hand_fused_names)):
        #f = KalmanFilter(dim_x=3, dim_z=3)
        #initial_state = np.zeros(3)
        #state_transition_mat = np.eye(3)
        #measurement_func = np.eye(3)
        #measurement_noise = np.eye(3) * 0.001
        #process_noise = [0.001] * 3
        #f.x = initial_state
        #f.F = state_transition_mat
        #f.H = measurement_func
        #f.P *= 1
        #f.R = measurement_noise
        #f.Q = process_noise
        #kalman_filters.append(f)

    while True:
        rs_hand_landmarks, rs_handedness = None, None
        oak_hand_landmarks, oak_handedness = None, None
        rs_arm_landmarks = None
        oak_arm_landmarks = None
        arm_fused_XYZ, hand_fused_XYZ = None, None 

        # 1. Get and preprocess rgb and its depth
        opposite_rgb = opposite_frame_getter_queue.get()
        rightside_rgb = rightside_frame_getter_queue.get()

        opposite_rgb = cv2.resize(opposite_rgb, frame_size)
        rightside_rgb = cv2.resize(rightside_rgb, frame_size)

        if arm_detection_activation:
            processed_rs_img = np.copy(opposite_rgb) 
            processed_oak_img = np.copy(rightside_rgb)
            if frame_color_format == "bgr":
                processed_rs_img = cv2.cvtColor(processed_rs_img, cv2.COLOR_BGR2RGB)
                processed_oak_img = cv2.cvtColor(processed_oak_img, cv2.COLOR_BGR2RGB)

            mp_rs_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=processed_rs_img)
            mp_oak_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=processed_oak_img)

            rs_arm_result = rs_arm_detector.detect_for_video(mp_rs_image, timestamp)
            oak_arm_result = oak_arm_detector.detect_for_video(mp_oak_image, timestamp)
            rs_arm_landmarks = get_normalized_pose_landmarks(rs_arm_result)
            oak_arm_landmarks = get_normalized_pose_landmarks(oak_arm_result)

        if hand_detection_activation:
            processed_rs_img = np.copy(opposite_rgb) 
            processed_oak_img = np.copy(rightside_rgb)
            if frame_color_format == "bgr":
                processed_rs_img = cv2.cvtColor(processed_rs_img, cv2.COLOR_BGR2RGB)
                processed_oak_img = cv2.cvtColor(processed_oak_img, cv2.COLOR_BGR2RGB)
            mp_rs_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=processed_rs_img)
            mp_oak_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=processed_oak_img)

            rs_hand_result = rs_hand_detector.detect_for_video(mp_rs_image, timestamp)
            oak_hand_result = oak_hand_detector.detect_for_video(mp_oak_image, timestamp)
            rs_hand_landmarks, rs_handedness = get_normalized_hand_landmarks(rs_hand_result)
            oak_hand_landmarks, oak_handedness = get_normalized_hand_landmarks(oak_hand_result)

        opposite_rgb = draw_arm_landmarks_on_image(opposite_rgb, rs_arm_landmarks)
        rightside_rgb = draw_arm_landmarks_on_image(rightside_rgb, oak_arm_landmarks)
        opposite_rgb = draw_hand_landmarks_on_image(opposite_rgb, rs_hand_landmarks, rs_handedness)
        rightside_rgb = draw_hand_landmarks_on_image(rightside_rgb, oak_hand_landmarks, oak_handedness)

        if rs_arm_landmarks and oak_arm_landmarks:
            if arm_num_pose_detected == 1:
                rs_arm_landmarks = rs_arm_landmarks[0]
                oak_arm_landmarks = oak_arm_landmarks[0]

            rs_arm_xyZ = get_xyZ(rs_arm_landmarks, 
                frame_size, 
                landmarks_id_want_to_get,
                arm_visibility_threshold)  # shape: (N, 3)
            oak_arm_xyZ = get_xyZ(oak_arm_landmarks, 
                frame_size, 
                landmarks_id_want_to_get,
                arm_visibility_threshold)  # shape: (N, 3)
                
            if (rs_arm_xyZ is not None and
                oak_arm_xyZ is not None and
                not np.isnan(rs_arm_xyZ).any() and
                not np.isnan(oak_arm_xyZ).any() and
                len(rs_arm_xyZ) == len(oak_arm_xyZ) and 
                len(rs_arm_xyZ) > 0 and
                len(oak_arm_xyZ) > 0):
                arm_fused_XYZ = fuse_landmarks_from_two_cameras(rs_arm_xyZ,  # shape: (N, 3)
                    oak_arm_xyZ,  
                    right_oak_ins,
                    left_oak_ins,
                    right_2_left_mat_avg)

        if (rs_hand_landmarks and 
            rs_handedness and 
            oak_hand_landmarks and 
            oak_handedness and
            hand_to_fuse in rs_handedness and
            hand_to_fuse in oak_handedness):

            rs_hand_id = rs_handedness.index(hand_to_fuse)
            oak_hand_id = oak_handedness.index(hand_to_fuse)
            rs_hand_landmarks = rs_hand_landmarks[rs_hand_id]
            oak_hand_landmarks = oak_hand_landmarks[oak_hand_id]

            rs_hand_xyZ = get_xyZ(rs_hand_landmarks,  # shape: (21, 3) for single hand
                frame_size,
                landmark_ids_to_get=None,
                visibility_threshold=None)
            oak_hand_xyZ = get_xyZ(oak_hand_landmarks,  # shape: (21, 3) for single hand
                frame_size,
                landmark_ids_to_get=None,
                visibility_threshold=None)

            if (rs_hand_xyZ is not None and
                oak_hand_xyZ is not None and 
                not np.isnan(rs_hand_xyZ).any() and
                not np.isnan(oak_hand_xyZ).any() and
                len(rs_hand_xyZ) == len(oak_hand_xyZ) and
                len(rs_hand_xyZ) > 0 and
                len(oak_hand_xyZ) > 0):
                hand_fused_XYZ = fuse_landmarks_from_two_cameras(rs_hand_xyZ,  # shape: (21, 3) for single hand
                    oak_hand_xyZ,
                    right_oak_ins,
                    left_oak_ins,
                    right_2_left_mat_avg)

        if (arm_fused_XYZ is not None and 
            hand_fused_XYZ is not None):
            """
            Convert these all to shoulder coord. to plot
            """
            arm_hand_fused_XYZ = np.concatenate([arm_fused_XYZ, hand_fused_XYZ], axis=0)
            arm_hand_XYZ_wrt_shoulder, xyz_origin = convert_to_shoulder_coord(arm_hand_fused_XYZ, arm_hand_fused_names)

            #for i in range(len(arm_hand_fused_names)):
                #kalman_filter = kalman_filters[i]
                #landmarks = arm_hand_XYZ_wrt_shoulder[i]
                #kalman_filter.predict()
                #kalman_filter.update(landmarks.flatten())
                #smooth_landmark = kalman_filter.x
                #arm_hand_XYZ_wrt_shoulder[i, :] = smooth_landmark.reshape(1, 3) 

            #angles = get_angles_between_joints(arm_hand_XYZ_wrt_shoulder, arm_hand_fused_names)

            if plot_3d:
                arm_points_vis_queue.put((arm_hand_XYZ_wrt_shoulder, xyz_origin))
                if arm_points_vis_queue.qsize() > 1:
                    arm_points_vis_queue.get()

        frame_count += 1
        elapsed_time = time.time() - start_time

        # Update FPS every second
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        timestamp += 1

        frame_of_two_cam = np.vstack([opposite_rgb, rightside_rgb])
        frame_of_two_cam = cv2.resize(frame_of_two_cam, (640, 720))
        cv2.putText(frame_of_two_cam, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        #rightside_rgb = cv2.resize(rightside_rgb, (960, 540))
        #cv2.putText(rightside_rgb, f'FPS: {fps:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4, cv2.LINE_AA)

        cv2.imshow("Frame", frame_of_two_cam)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break