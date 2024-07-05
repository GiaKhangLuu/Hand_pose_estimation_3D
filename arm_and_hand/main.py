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

from mediapipe_drawing import draw_hand_landmarks_on_image, draw_arm_landmarks_on_image
from camera_tools import initialize_oak_cam, initialize_realsense_cam, stream_rs, stream_oak
from utilities import (filter_depth,
    detect_hand_landmarks,
    detect_arm_landmarks)
from common import (load_data_from_npz_file,
    get_oak_2_rs_matrix,
    load_config)

# ------------------- READ CONFIG ------------------- 
config_file_path = os.path.join(CURRENT_DIR, "configuration.yaml") 
config = load_config(config_file_path)

realsense_rgb_size = (config["camera"]["realsense"]["rgb"]["width"], 
    config["camera"]["realsense"]["rgb"]["height"])
realsense_depth_size = (config["camera"]["realsense"]["depth"]["width"], 
    config["camera"]["realsense"]["depth"]["height"]) 
opposite_cam_calib_path = config["camera"]["realsense"]["opposite_camera_calibration_path"]

oak_stereo_size = (config["camera"]["oak"]["stereo"]["width"], 
    config["camera"]["oak"]["stereo"]["height"]) 
rightside_cam_calib_path = config["camera"]["oak"]["rightside_camera_calibration_path"]

frame_size = (config["process_frame"]["frame_size"]["width"], 
    config["process_frame"]["frame_size"]["height"])
sliding_window_size = config["process_frame"]["depth_smoothing"]["sliding_window_size"]
sigma_color =  config["process_frame"]["depth_smoothing"]["sigma_color"]
sigma_space = config["process_frame"]["depth_smoothing"]["sigma_space"]       
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
hand_visibility_threshold = config["hand_landmark_detection"]["visibility_threshold"]
hand_num_hand_detected = config["hand_landmark_detection"]["num_hand"]
hand_model_path = config["hand_landmark_detection"]["model_asset_path"]
hand_fuse_left = config["hand_landmark_detection"]["hand_to_fuse"]["left"]
hand_fuse_right = config["hand_landmark_detection"]["hand_to_fuse"]["right"]
hand_fuse_both = config["hand_landmark_detection"]["hand_to_fuse"]["both"]

# -------------------- GET TRANSFORMATION MATRIX -------------------- 
oak_data = load_data_from_npz_file(rightside_cam_calib_path)
rs_data = load_data_from_npz_file(opposite_cam_calib_path)

oak_r_raw, oak_t_raw, oak_ins = oak_data['rvecs'], oak_data['tvecs'], oak_data['camMatrix']
rs_r_raw, rs_t_raw, rs_ins = rs_data['rvecs'], rs_data['tvecs'], rs_data['camMatrix']

oak_2_rs_mat_avg = get_oak_2_rs_matrix(oak_r_raw, oak_t_raw, 
    rs_r_raw, rs_t_raw)

# -------------------- INIT MEDIAPIPE MODELS -------------------- 
hand_base_options = python.BaseOptions(model_asset_path=hand_model_path,
    delegate=mp.tasks.BaseOptions.Delegate.GPU)
hand_options = vision.HandLandmarkerOptions(
    base_options=hand_base_options,
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=hand_num_hand_detected,
    min_hand_detection_confidence=hand_min_det_conf,
    min_tracking_confidence=hand_min_tracking_conf)
rs_hand_detector = vision.HandLandmarker.create_from_options(hand_options)
oak_hand_detector = vision.HandLandmarker.create_from_options(hand_options)

arm_base_options = python.BaseOptions(model_asset_path=arm_model_path,
    delegate=mp.tasks.BaseOptions.Delegate.GPU)
arm_options = vision.PoseLandmarkerOptions(
    base_options=arm_base_options,
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_poses=arm_num_pose_detected,
    min_pose_detection_confidence=arm_min_det_conf,
    min_tracking_confidence=arm_min_tracking_conf,
    output_segmentation_masks=False)
rs_arm_detector = vision.PoseLandmarker.create_from_options(arm_options)  # Each camera is responsible for one detector because the detector stores the pose in prev. frame
oak_arm_detector = vision.PoseLandmarker.create_from_options(arm_options)  # Each camera is responsible for one detector because the detector stores the pose in prev. frame

# -------------------- CREATE QUEUE TO INTERACT WITH THREADS -------------------- 
opposite_frame_getter_queue = queue.Queue()  # This queue directly gets RAW frame (rgb, depth) from REALSENSE camera
rightside_frame_getter_queue = queue.Queue()  # This queue directly gets RAW frame (rgb, depth) from OAK camera

hand_landmark_input_queue = queue.Queue()  # This queue stores couple input rgb images (rs and oak) for HAND landmark detector
hand_landmark_detection_result_queue = queue.Queue()  # This queue stores couple detection results of HAND landmarks

arm_landmark_input_queue = queue.Queue()  # This queue stores couple input rgb images (rs and oak) for ARM landmark detector
arm_landmark_detection_result_queue = queue.Queue()  # This queue stores couple detection results of ARM landmarks

if __name__ == "__main__":
    pipeline_rs, rsalign = initialize_realsense_cam(realsense_rgb_size, realsense_depth_size)
    pipeline_oak = initialize_oak_cam(oak_stereo_size)

    rs_thread = threading.Thread(target=stream_rs, args=(pipeline_rs, rsalign, 
        opposite_frame_getter_queue,), daemon=True)

    oak_thread = threading.Thread(target=stream_oak, args=(pipeline_oak,  
        rightside_frame_getter_queue,), daemon=True)

    hand_landmark_detection_thread = threading.Thread(target=detect_hand_landmarks, 
        args=(rs_hand_detector, 
            oak_hand_detector,
            hand_landmark_input_queue,
            hand_landmark_detection_result_queue,
            frame_color_format), 
        daemon=True)

    arm_landmark_detection_thread = threading.Thread(target=detect_arm_landmarks, 
        args=(rs_arm_detector, 
            oak_arm_detector,
            arm_landmark_input_queue,
            arm_landmark_detection_result_queue,
            frame_color_format), 
        daemon=True)

    rs_thread.start()
    oak_thread.start()
    if hand_detection_activation:
        hand_landmark_detection_thread.start()
    if arm_detection_activation:
        arm_landmark_detection_thread.start()

    timestamp = 0
    frame_count = 0
    start_time = time.time()
    fps = 0

    while True:
        # 1. Preprocess rgb and its depth
        opposite_rgb, opposite_depth = opposite_frame_getter_queue.get()
        rightside_rgb, rightside_depth = rightside_frame_getter_queue.get()

        opposite_rgb = cv2.resize(opposite_rgb, frame_size)
        opposite_depth = cv2.resize(opposite_depth, frame_size)
        rightside_rgb = cv2.resize(rightside_rgb, frame_size)
        rightside_depth = cv2.resize(rightside_depth, frame_size)

        assert opposite_depth.shape == opposite_rgb.shape[:-1]
        assert rightside_depth.shape == rightside_rgb.shape[:-1]
        
        # 2.1. Detect arm landmarks
        arm_landmark_input_queue.put((np.copy(opposite_rgb), np.copy(rightside_rgb), timestamp))

        time.sleep(0.005)

        if not arm_landmark_detection_result_queue.empty():
            rs_arm_landmarks_result, oak_arm_landmarks_result = arm_landmark_detection_result_queue.get()
            opposite_rgb = draw_arm_landmarks_on_image(opposite_rgb, rs_arm_landmarks_result)
            rightside_rgb = draw_arm_landmarks_on_image(rightside_rgb, oak_arm_landmarks_result)

        # 2.2. Detect hand landmarks
        hand_landmark_input_queue.put((np.copy(opposite_rgb), np.copy(rightside_rgb), timestamp))
        timestamp += 1

        # --------------- WAITING FOR THE MODELS TO PROCESS --------------- 
        time.sleep(0.005)

        if not hand_landmark_detection_result_queue.empty():
            rs_hand_landmarks_result, oak_hand_landmarks_result = hand_landmark_detection_result_queue.get()
            opposite_rgb = draw_hand_landmarks_on_image(opposite_rgb, rs_hand_landmarks_result)
            rightside_rgb = draw_hand_landmarks_on_image(rightside_rgb, oak_hand_landmarks_result)



        # 3. Get xyZ
        # ....

        frame_count += 1
        elapsed_time = time.time() - start_time

        # Update FPS every second
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        frame_of_two_cam = np.vstack([opposite_rgb, rightside_rgb])
        frame_of_two_cam = cv2.resize(frame_of_two_cam, (640, 720))
        cv2.putText(frame_of_two_cam, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("Frame", frame_of_two_cam)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            pipeline_rs.stop()
            break