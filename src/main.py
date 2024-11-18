"""
Convention:
    rs = opposite = REALSENSE camera = left
    oak = rightside = OAK camera = right
"""

import sys
import os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, '..'))

import yaml
import cv2
import numpy as np
import queue
import threading
import time
import shutil
import socket
import math
from datetime import datetime
from pathlib import Path

from stream_3d import visualize_sticky_man
from camera_tools import initialize_oak_cam, initialize_realsense_cam, stream_rs, stream_oak
from csv_writer import (
    create_csv, 
    append_to_csv, 
    fusion_csv_columns_name, 
    split_train_test_val,
    left_arm_angles_name,
    right_arm_angles_name,
    left_hand_angles_name,
    timestamp_columns 
)
from utilities import (convert_to_left_shoulder_coord,
    flatten_two_camera_input)
from common import (load_data_from_npz_file,
    get_oak_2_rs_matrix,
    load_config, 
    scale_intrinsic_by_res)
from send_angles_to_robot import send_angles_to_robot_using_pid
from landmark_detector import RTMPoseDetector_BothSides
from landmarks_fuser import LandmarksFuser
from angle_smoother import ArmAngleSmoother, HandAngleSmoother
from angle_calculator import (
    HeadAngleCalculator,
    LeftArmAngleCalculator,
    RightArmAngleCalculator,
    LeftHandAngleCalculator
)

# ------------------- READ CONFIG ------------------- 
WORK_DIR = Path(os.environ["MOCAP_WORKDIR"])
MAIN_CONFIG_FILE_PATH = os.path.join(CURRENT_DIR, "configuration", "main_conf.yaml") 
config = load_config(MAIN_CONFIG_FILE_PATH)

DETECTION_CONFIG_FILE_PATH = os.path.join(CURRENT_DIR, "configuration", "detection_config.yaml")
detection_config = load_config(DETECTION_CONFIG_FILE_PATH)

left_oak_mxid = config["camera"]["left_camera"]["mxid"]
left_cam_calib_path = WORK_DIR / config["camera"]["left_camera"]["left_camera_calibration_path"]
right_oak_mxid = config["camera"]["right_camera"]["mxid"]
right_cam_calib_path = WORK_DIR / config["camera"]["right_camera"]["right_camera_calibration_path"]

frame_size = (config["process_frame"]["frame_size"]["width"], 
    config["process_frame"]["frame_size"]["height"])
frame_calibrated_size = (config["camera"]["frame_calibrated_size"]["height"],
    config["camera"]["frame_calibrated_size"]["width"])

send_udp = config["send_udp"]
plot_3d = config["utilities"]["plot_3d"]
draw_landmarks = config["utilities"]["draw_landmarks"]
save_timestamp = config["utilities"]["save_timestamp"]
save_landmarks = config["utilities"]["save_landmarks"]
save_images = config["utilities"]["save_images"]
save_depth = config["utilities"]["save_depth"]
save_left_arm_angles = config["utilities"]["save_left_arm_angles"]
save_right_arm_angles = config["utilities"]["save_right_arm_angles"]
save_left_hand_angles = config["utilities"]["save_left_hand_angles"]
is_head_fused = config["utilities"]["fusing_head"]
is_left_arm_fused = config["utilities"]["fusing_left_arm"]
is_left_hand_fused = config["utilities"]["fusing_left_hand"]
is_right_arm_fused = config["utilities"]["fusing_right_arm"]
is_right_hand_fused = config["utilities"]["fusing_right_hand"]

detection_model_selection_id = detection_config["model_selection_id"]
detection_model_list = tuple(detection_config["model_list"])

head_fused_names = config["fusing_head_landmark_dictionary"]
arm_hand_fused_names = config["fusing_body_landmark_dictionary"]
fusing_landmark_dictionary = arm_hand_fused_names.copy()
# This dictionary specifies which landmarks we will get after detecting phase.
# Note that this value must be match with the `selected_landmarks_idx` in `detection_config.yaml`.
fusing_landmark_dictionary.extend(head_fused_names)  

fusing_enable = config["fusing_phase"]["enable"]
fusing_method_list = tuple(config["fusing_phase"]["fusing_methods"])
fusing_method_selection_id = config["fusing_phase"]["fusing_selection_id"]

reduce_noise_config = config["reduce_noise_phase"]
enable_left_arm_angles_noise_reducer = reduce_noise_config["left_arm_angles_noise_reducer"]["enable"]
left_arm_angles_noise_statistical_file = WORK_DIR / reduce_noise_config["left_arm_angles_noise_reducer"]["statistical_file"]
left_arm_angles_noise_reducer_dim = reduce_noise_config["left_arm_angles_noise_reducer"]["dim"]
enable_right_arm_angles_noise_reducer = reduce_noise_config["right_arm_angles_noise_reducer"]["enable"]
right_arm_angles_noise_statistical_file = WORK_DIR / reduce_noise_config["right_arm_angles_noise_reducer"]["statistical_file"]
right_arm_angles_noise_reducer_dim = reduce_noise_config["right_arm_angles_noise_reducer"]["dim"]
enable_left_hand_angles_noise_reducer = reduce_noise_config["left_hand_angles_noise_reducer"]["enable"]
left_hand_angles_noise_statistical_file = WORK_DIR / reduce_noise_config["left_hand_angles_noise_reducer"]["statistical_file"]
left_hand_angles_noise_reducer_dim = reduce_noise_config["left_hand_angles_noise_reducer"]["dim"]

draw_xyz = config["debugging_mode"]["draw_xyz"]
draw_left_arm_coordinates = config["debugging_mode"]["show_left_arm"]
draw_left_hand_coordinates = config["debugging_mode"]["show_left_hand"]
ref_vector_color = list(config["debugging_mode"]["ref_vector_color"])
joint_vector_color = list(config["debugging_mode"]["joint_vector_color"])

run_to_collect_data = config["run_to_collect_data_for_fusing_and_detection"]

if is_left_hand_fused:
    is_left_arm_fused = True 
if is_right_hand_fused:
    is_right_arm_fused = True
        
LANDMARKS_TO_FUSED = ["left shoulder", "left hip", "right shoulder", "right hip"]
LANDMARKS_TO_FUSED_IDX = []  # This list specifies which landmarks will be fused from two cameras. Landmarks which are not specified to fuse will have 0 values.
if is_left_arm_fused:
    LEFT_ARM_LANDMARKS_TO_FUSED = ["left elbow", "WRIST", "INDEX_FINGER_MCP", "MIDDLE_FINGER_MCP"]
    LANDMARKS_TO_FUSED.extend(LEFT_ARM_LANDMARKS_TO_FUSED)
if is_left_hand_fused:            
    LEFT_HAND_LANDMARKS_TO_FUSED = ["THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",    
                                    "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",    
                                    "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",    
                                    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",    
                                    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"]
    LANDMARKS_TO_FUSED.extend(LEFT_HAND_LANDMARKS_TO_FUSED)
if is_right_arm_fused:
    RIGHT_ARM_LANDMARKS_TO_FUSED = ["right elbow", "RIGHT_WRIST", "RIGHT_INDEX_FINGER_MCP", "RIGHT_MIDDLE_FINGER_MCP"]
    LANDMARKS_TO_FUSED.extend(RIGHT_ARM_LANDMARKS_TO_FUSED)                 
if is_right_hand_fused:
    RIGHT_HAND_LANDMARKS_TO_FUSED = ["RIGHT_THUMB_CMC", "RIGHT_THUMB_MCP", "RIGHT_THUMB_IP", "RIGHT_THUMB_TIP",    
                                    "RIGHT_INDEX_FINGER_PIP", "RIGHT_INDEX_FINGER_DIP", "RIGHT_INDEX_FINGER_TIP",    
                                    "RIGHT_MIDDLE_FINGER_PIP", "RIGHT_MIDDLE_FINGER_DIP", "RIGHT_MIDDLE_FINGER_TIP",    
                                    "RIGHT_RING_FINGER_MCP", "RIGHT_RING_FINGER_PIP", "RIGHT_RING_FINGER_DIP", "RIGHT_RING_FINGER_TIP",    
                                    "RIGHT_PINKY_MCP", "RIGHT_PINKY_PIP", "RIGHT_PINKY_DIP", "RIGHT_PINKY_TIP"]
    LANDMARKS_TO_FUSED.extend(RIGHT_HAND_LANDMARKS_TO_FUSED)
if is_head_fused:
    HEAD_LANDMARKS_TO_FUSED = ["nose", "left eye", "right eye", "left ear", "right ear"]
    LANDMARKS_TO_FUSED.extend(HEAD_LANDMARKS_TO_FUSED)
            
for landmark_name in LANDMARKS_TO_FUSED:
    LANDMARKS_TO_FUSED_IDX.append(fusing_landmark_dictionary.index(landmark_name)) 

if run_to_collect_data:
    plot_3d = True
    send_udp = False
    save_landmarks = True
    save_images = True
    fusing_enable = True
    fusing_method_selection_id = 1
    config["fusing_phase"]["minimize_distance"]["tolerance"] = 1e-5

# -------------------- GET TRANSFORMATION MATRIX -------------------- 
right_cam_data = load_data_from_npz_file(right_cam_calib_path)
left_cam_data = load_data_from_npz_file(left_cam_calib_path)

right_oak_r_raw, right_oak_t_raw, right_camera_intr = right_cam_data['rvecs'], right_cam_data['tvecs'], right_cam_data['camMatrix']
left_oak_r_raw, left_oak_t_raw, left_camera_intr = left_cam_data['rvecs'], left_cam_data['tvecs'], left_cam_data['camMatrix']

right_camera_intr = scale_intrinsic_by_res(right_camera_intr, frame_calibrated_size, frame_size[::-1])
left_camera_intr = scale_intrinsic_by_res(left_camera_intr, frame_calibrated_size, frame_size[::-1])

right_2_left_mat_avg = get_oak_2_rs_matrix(right_oak_r_raw, right_oak_t_raw, 
    left_oak_r_raw, left_oak_t_raw)

# -------------------- CREATE QUEUE TO INTERACT WITH THREADS -------------------- 
left_frame_getter_queue = queue.Queue()  # This queue directly gets RAW frame (rgb, depth) from REALSENSE camera
right_frame_getter_queue = queue.Queue()  # This queue directly gets RAW frame (rgb, depth) from OAK camera

arm_points_vis_queue = queue.Queue()  # This queue stores fused arm landmarks to visualize by open3d
hand_points_vis_queue = queue.Queue()  # This queue stores fused hand landmarks to visualize by open3d 
TARGET_LEFT_ARM_HAND_ANGLES_QUEUE = queue.Queue()  # This queue stores target angles 

landmarks_detectors = RTMPoseDetector_BothSides(
    model_config=detection_config["mmpose"],
    draw_landmarks=draw_landmarks)
landmarks_fuser = LandmarksFuser(
    method_selected_id=fusing_method_selection_id,
    method_list=fusing_method_list, 
    method_config=config["fusing_phase"], 
    img_size=frame_size)
if enable_left_arm_angles_noise_reducer:
    left_arm_angle_noise_reducer = ArmAngleSmoother(
        angles_noise_statistical_file=left_arm_angles_noise_statistical_file, 
        dim=left_arm_angles_noise_reducer_dim)
if enable_right_arm_angles_noise_reducer:
    right_arm_angle_noise_reducer = ArmAngleSmoother(
        angles_noise_statistical_file=right_arm_angles_noise_statistical_file,
        dim=right_arm_angles_noise_reducer_dim)
if enable_left_hand_angles_noise_reducer:
    left_hand_angle_noise_reducer = HandAngleSmoother(
        angles_noise_statistical_file=left_hand_angles_noise_statistical_file,
        dim=left_hand_angles_noise_reducer_dim)

head_angle_calculator = HeadAngleCalculator(num_chain=1, landmark_dictionary=fusing_landmark_dictionary)
left_arm_angle_calculator = LeftArmAngleCalculator(num_chain=3, landmark_dictionary=fusing_landmark_dictionary)
right_arm_angle_calculator = RightArmAngleCalculator(num_chain=3, landmark_dictionary=fusing_landmark_dictionary)
left_hand_angle_calculator = LeftHandAngleCalculator(num_chain=2, landmark_dictionary=fusing_landmark_dictionary)

DATA_DIR = None
LANDMARK_CSV_PATH = None
IMAGE_DIR = None
LEFT_ARM_ANGLE_CSV_PATH = None
RIGHT_ARM_ANGLE_CSV_PATH = None
LEFT_HAND_ANGLE_CSV_PATH = None
TIMESTAMP_PATH = None
DEPTH_DIR = None

WAIT_TIME_TO_STAND_IN_POSITION = 1

def create_files(
    save_landmarks, 
    save_images, 
    save_left_arm_angles,
    save_right_arm_angles, 
    save_left_hand_angles,
    save_timestamp,
    save_depth
):
    global DATA_DIR
    global LANDMARK_CSV_PATH
    global IMAGE_DIR
    global LEFT_ARM_ANGLE_CSV_PATH
    global RIGHT_ARM_ANGLE_CSV_PATH
    global LEFT_HAND_ANGLE_CSV_PATH
    global TIMESTAMP_PATH
    global DEPTH_DIR

    current_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    current_date = datetime.now().strftime('%Y-%m-%d')
    DATA_DIR = WORK_DIR / os.path.join(CURRENT_DIR, "../..", "data", current_date)
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    DATA_DIR = os.path.join(DATA_DIR, current_time)
    os.mkdir(DATA_DIR)
        
    if save_landmarks:
        LANDMARK_CSV_PATH = os.path.join(DATA_DIR, "landmarks_all_{}.csv".format(current_time))
        create_csv(LANDMARK_CSV_PATH, fusion_csv_columns_name)

    if save_images:
        IMAGE_DIR = os.path.join(DATA_DIR, "image") 
        os.mkdir(IMAGE_DIR)

    if save_left_arm_angles:
        LEFT_ARM_ANGLE_CSV_PATH = os.path.join(DATA_DIR, "left_arm_angle_{}.csv".format(current_time))
        create_csv(LEFT_ARM_ANGLE_CSV_PATH, left_arm_angles_name)
            
    if save_right_arm_angles:
        RIGHT_ARM_ANGLE_CSV_PATH = os.path.join(DATA_DIR, "right_arm_angle_{}.csv".format(current_time))
        create_csv(RIGHT_ARM_ANGLE_CSV_PATH, right_arm_angles_name)
            
    if save_left_hand_angles:
        LEFT_HAND_ANGLE_CSV_PATH = os.path.join(DATA_DIR, "left_hand_angle_{}.csv".format(current_time))
        create_csv(LEFT_HAND_ANGLE_CSV_PATH, left_hand_angles_name)

    if save_timestamp:
        TIMESTAMP_PATH = os.path.join(DATA_DIR, "timestamp_{}.csv".format(current_time))
        create_csv(TIMESTAMP_PATH, timestamp_columns)
    
    if save_depth:
        DEPTH_DIR = os.path.join(DATA_DIR, "depth")
        os.mkdir(DEPTH_DIR)

def add_timestamp(timestamp_list):
    timestamp_list.append(time.time())

if __name__ == "__main__":

    RAW_LEFT_ARM_ANGLES_COLUMN_NAMES = ["time", "raw_joint1", "raw_joint2", "raw_joint3", "raw_joint4", "raw_joint5", "raw_joint6"]
    RAW_LEFT_ARM_ANGLE_CSV_FILE = "/home/giakhang/Desktop/raw_debug_left_arm_angle.csv"

    create_csv(RAW_LEFT_ARM_ANGLE_CSV_FILE, RAW_LEFT_ARM_ANGLES_COLUMN_NAMES)

    time.sleep(WAIT_TIME_TO_STAND_IN_POSITION)

    create_dir = (
        save_landmarks or 
        save_left_arm_angles or 
        save_right_arm_angles or
        save_left_hand_angles or
        save_images or 
        save_timestamp or
        save_depth
    )

    if create_dir:
        create_files(
            save_landmarks=save_landmarks, 
            save_images=save_images,
            save_left_arm_angles=save_left_arm_angles,
            save_right_arm_angles=save_right_arm_angles, 
            save_left_hand_angles=save_left_hand_angles,
            save_timestamp=save_timestamp,
            save_depth=save_depth
        )

    pipeline_left_oak = initialize_oak_cam()
    pipeline_right_oak = initialize_oak_cam()

    left_oak_thread = threading.Thread(target=stream_oak, args=(pipeline_left_oak,  
        left_frame_getter_queue, left_oak_mxid), daemon=True)
    right_oak_thread = threading.Thread(target=stream_oak, args=(pipeline_right_oak,  
        right_frame_getter_queue, right_oak_mxid), daemon=True)

    left_oak_thread.start()
    right_oak_thread.start()

    if send_udp:
        send_left_arm_hand_data_thread = threading.Thread(target=send_angles_to_robot_using_pid, 
            args=(TARGET_LEFT_ARM_HAND_ANGLES_QUEUE, True,), daemon=True)
        send_left_arm_hand_data_thread.start()

    if plot_3d:
        vis_arm_thread = threading.Thread(target=visualize_sticky_man,
            args=(arm_points_vis_queue, 
                fusing_landmark_dictionary,
                LANDMARKS_TO_FUSED_IDX,
                left_arm_angle_calculator,
                left_hand_angle_calculator,
                draw_xyz,
                draw_left_arm_coordinates,
                draw_left_hand_coordinates,
                joint_vector_color,
                ref_vector_color,
                is_head_fused,
                is_left_arm_fused,
                is_left_hand_fused,
                is_right_arm_fused,
                is_right_hand_fused),
            daemon=True)
        vis_arm_thread.start()

    timestamp = 0
    frame_count = 0
    num_img_save = 0
    every_1k_timestamp = 1
    start_time = time.time()
    fps = 0

    while True:
        timestamp_each_stage = [timestamp]
        left_camera_body_landmarks_xyZ, right_camera_body_landmarks_xyZ = None, None
        succeed_in_fusing_landmarks = False

        # Get and preprocess rgb and its depth
        left_rgb, left_depth = left_frame_getter_queue.get()
        right_rgb, right_depth = right_frame_getter_queue.get()

        add_timestamp(timestamp_each_stage)

        left_rgb = cv2.resize(left_rgb, frame_size)
        right_rgb = cv2.resize(right_rgb, frame_size)
        left_depth = cv2.resize(left_depth, frame_size)
        right_depth = cv2.resize(right_depth, frame_size)

        left_depth_to_save = np.copy(left_depth)
        right_depth_to_save = np.copy(right_depth)
        left_image_to_save = np.copy(left_rgb)
        right_image_to_save = np.copy(right_rgb)

        # Detection phase
        detection_input = {
            "left_cam_rgb": left_rgb,
            "left_cam_depth": left_depth,
            "right_cam_rgb":right_rgb,
            "right_cam_depth": right_depth,
        } 

        detection_result = landmarks_detectors(detection_input)

        left_camera_body_landmarks_xyZ = detection_result["left_cam_detection_result"]
        right_camera_body_landmarks_xyZ = detection_result["right_cam_detection_result"]
        if draw_landmarks:
            left_rgb = detection_result["left_cam_drawed_img"]
            right_rgb = detection_result["right_cam_drawed_img"]

        add_timestamp(timestamp_each_stage)

        #time.sleep(1)

        # Fusing phase
        if (fusing_enable and
            left_camera_body_landmarks_xyZ is not None and 
            right_camera_body_landmarks_xyZ is not None and
            left_camera_body_landmarks_xyZ.shape[0] == len(fusing_landmark_dictionary) and
            left_camera_body_landmarks_xyZ.shape[0] == right_camera_body_landmarks_xyZ.shape[0]):
            
            left_xyZ_to_fused = np.zeros_like(left_camera_body_landmarks_xyZ, dtype=np.float64)
            right_xyZ_to_fused = np.zeros_like(right_camera_body_landmarks_xyZ, dtype=np.float64)
            
            left_xyZ_to_fused[LANDMARKS_TO_FUSED_IDX] = left_camera_body_landmarks_xyZ[LANDMARKS_TO_FUSED_IDX]
            right_xyZ_to_fused[LANDMARKS_TO_FUSED_IDX] = right_camera_body_landmarks_xyZ[LANDMARKS_TO_FUSED_IDX]
            
            arm_hand_fused_XYZ = landmarks_fuser.fuse(
                left_camera_wholebody_xyZ=left_xyZ_to_fused,
                right_camera_wholebody_xyZ=right_xyZ_to_fused,
                left_camera_intr=left_camera_intr,
                right_camera_intr=right_camera_intr,
                right_2_left_matrix=right_2_left_mat_avg)

            add_timestamp(timestamp_each_stage)

            arm_hand_XYZ_wrt_left_shoulder, xyz_origin = convert_to_left_shoulder_coord(
                arm_hand_fused_XYZ,
                fusing_landmark_dictionary)
            
            if (arm_hand_XYZ_wrt_left_shoulder is not None and
                xyz_origin is not None):

                if is_head_fused:
                    head_result = head_angle_calculator(arm_hand_XYZ_wrt_left_shoulder,
                        parent_coordinate=xyz_origin)
                    head_angles = head_result["head"]["angles"]
                    #print(head_angles)
                else:
                    head_angles = [0] * 2

                if is_left_arm_fused:
                    left_arm_result = left_arm_angle_calculator(arm_hand_XYZ_wrt_left_shoulder, 
                        parent_coordinate=xyz_origin)
                    left_arm_angles = left_arm_result["left_arm"]["angles"]
                    left_arm_rot_mats_wrt_origin = left_arm_result["left_arm"]["rot_mats_wrt_origin"]
                    last_coordinate_from_left_arm = left_arm_rot_mats_wrt_origin[-1]
                    
                    raw_left_arm_angles = [*left_arm_angles]
                    if enable_left_arm_angles_noise_reducer:
                        left_arm_angles = np.array(left_arm_angles)
                        left_arm_angles = left_arm_angle_noise_reducer(left_arm_angles)
                    smooth_left_arm_angles = [*left_arm_angles]
                else:
                    last_coordinate_from_left_arm = None
                    left_arm_result = None
                    left_arm_angles = raw_left_arm_angles = smooth_left_arm_angles = [0] * 6 
                    
                if is_right_arm_fused:
                    right_arm_result = right_arm_angle_calculator(arm_hand_XYZ_wrt_left_shoulder,
                        parent_coordinate=xyz_origin)
                    right_arm_angles = right_arm_result["right_arm"]["angles"]
                    right_arm_rot_mats_wrt_origin = right_arm_result["right_arm"]["rot_mats_wrt_origin"]
                    last_coordinate_from_right_arm = right_arm_rot_mats_wrt_origin[-1]

                    raw_right_arm_angles = [*right_arm_angles]
                    if enable_right_arm_angles_noise_reducer:
                        right_arm_angles = np.array(right_arm_angles)
                        right_arm_angles = right_arm_angle_noise_reducer(right_arm_angles)
                    smooth_right_arm_angles = [*right_arm_angles]
                else:
                    right_arm_result = None
                    right_arm_angles = raw_right_arm_angles = smooth_right_arm_angles = [0] * 6
                    
                if is_left_hand_fused:
                    left_hand_result = left_hand_angle_calculator(arm_hand_XYZ_wrt_left_shoulder, 
                        parent_coordinate=last_coordinate_from_left_arm)
                    left_hand_angles = []
                    for finger_name in left_hand_angle_calculator.fingers_name:
                        finger_i_angles = left_hand_result[finger_name]["angles"].copy()

                        # In robot, its finger joint 1 is our finger joint 2, and vice versa (EXCEPT FOR THUMB FINGER). 
                        # So that, we need to swap these values.
                        if finger_name != "THUMB":
                            finger_i_angles[0], finger_i_angles[1] = finger_i_angles[1], finger_i_angles[0]

                        left_hand_angles.extend(finger_i_angles)
                    
                    raw_left_hand_angles = [*left_hand_angles] 
                    if enable_left_hand_angles_noise_reducer:    
                        left_hand_angles = np.array(left_hand_angles)
                        left_hand_angles = left_hand_angle_noise_reducer(left_hand_angles)
                    smooth_left_hand_angles = [*left_hand_angles]
                else:
                    left_hand_result = None
                    left_hand_angles = raw_left_hand_angles = smooth_left_hand_angles = [0] * 15
                    
                succeed_in_fusing_landmarks = True

                head_angles_to_send = np.array(head_angles) 
                left_arm_angles_to_send = np.array(left_arm_angles)
                right_arm_angles_to_send = np.array(right_arm_angles)
                left_hand_angles_to_send = np.array(left_hand_angles)

                append_to_csv(
                    RAW_LEFT_ARM_ANGLE_CSV_FILE, 
                    [time.time(), *left_arm_angles_to_send]
                )

                add_timestamp(timestamp_each_stage)

                angles_sent_to_robot = {
                    "head": head_angles_to_send,
                    "left_arm": left_arm_angles_to_send,
                    "right_arm": right_arm_angles_to_send,
                    "left_fingers": left_hand_angles_to_send
                }
                
                if send_udp:
                    TARGET_LEFT_ARM_HAND_ANGLES_QUEUE.put((angles_sent_to_robot, timestamp))
                    if TARGET_LEFT_ARM_HAND_ANGLES_QUEUE.qsize() > 1:
                        TARGET_LEFT_ARM_HAND_ANGLES_QUEUE.get()

                add_timestamp(timestamp_each_stage)

                if plot_3d: 
                    lmks_and_angle_result = (arm_hand_XYZ_wrt_left_shoulder, xyz_origin, left_arm_result, left_hand_result)
                    arm_points_vis_queue.put(lmks_and_angle_result)
                    if arm_points_vis_queue.qsize() > 1:
                        arm_points_vis_queue.get()

                if save_landmarks and timestamp > 5:  # warm up for 100 frames before saving landmarks
                    left_input = left_camera_body_landmarks_xyZ[:48]  # For now, just get `left arm-hand` and `right arm-hand` (havent get `head` yet).
                    right_input = right_camera_body_landmarks_xyZ[:48]  # For now, just get `left arm-hand` and `right arm-hand` (havent get `head` yet).
                    input_row = flatten_two_camera_input(
                        left_camera_landmarks_xyZ=left_input,
                        right_camera_landmarks_xyZ=right_input,
                        left_camera_intr=left_camera_intr,
                        right_camera_intr=right_camera_intr,
                        right_2_left_matrix=right_2_left_mat_avg,
                        img_size=frame_size,
                        frame_calibrated_size=frame_calibrated_size[::-1], 
                        mode="ground_truth",
                        timestamp=timestamp,
                        output_landmarks=arm_hand_fused_XYZ[:48])
                    append_to_csv(LANDMARK_CSV_PATH, input_row)

        if save_depth and timestamp > 5:
            left_depth_path = os.path.join(DEPTH_DIR, "left_{}.npy".format(timestamp))
            right_depth_path = os.path.join(DEPTH_DIR, "right_{}.npy".format(timestamp))
            np.save(left_depth_path, left_depth_to_save)
            np.save(right_depth_path, right_depth_to_save)
            
        if save_images and timestamp > 5:  # warm up for 100 frames before saving image 
            left_img_path = os.path.join(IMAGE_DIR, "left_{}.jpg".format(timestamp))
            right_img_path = os.path.join(IMAGE_DIR, "right_{}.jpg".format(timestamp))
            cv2.imwrite(left_img_path, left_image_to_save)
            cv2.imwrite(right_img_path, right_image_to_save)
            num_img_save += 1

        frame_count += 1
        elapsed_time = time.time() - start_time

        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        frame_of_two_cam = np.vstack([left_rgb, right_rgb])
        frame_of_two_cam = cv2.resize(frame_of_two_cam, (640, 720))
        cv2.putText(frame_of_two_cam, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("Frame", frame_of_two_cam)

        if save_timestamp and succeed_in_fusing_landmarks:
            append_to_csv(TIMESTAMP_PATH, timestamp_each_stage)
        
        if save_left_arm_angles and succeed_in_fusing_landmarks:
            row_left_arm_angles_writed_to_file = [timestamp, 
                                                  round(fps, 1), 
                                                  *raw_left_arm_angles,
                                                  *smooth_left_arm_angles]
            append_to_csv(LEFT_ARM_ANGLE_CSV_PATH, row_left_arm_angles_writed_to_file)
            
        if save_right_arm_angles and succeed_in_fusing_landmarks:
            row_right_arm_angles_writed_to_file = [timestamp,
                                                   round(fps, 1), 
                                                   *raw_right_arm_angles,
                                                   *smooth_right_arm_angles]
            append_to_csv(RIGHT_ARM_ANGLE_CSV_PATH, row_right_arm_angles_writed_to_file)
                    
        if save_left_hand_angles and succeed_in_fusing_landmarks:
            row_left_hand_angles_writed_to_file = [timestamp,
                                                   round(fps, 1),
                                                   *raw_left_hand_angles,
                                                   *smooth_left_hand_angles] 
            append_to_csv(LEFT_HAND_ANGLE_CSV_PATH, row_left_hand_angles_writed_to_file)

        timestamp += 1
        
        # If timestamp 
        if timestamp // 1000 >= every_1k_timestamp and create_dir:
            every_1k_timestamp += 1
            if save_landmarks:
                split_train_test_val(LANDMARK_CSV_PATH)
            create_files(
                save_landmarks=save_landmarks, 
                save_images=save_images,
                save_left_arm_angles=save_left_arm_angles,
                save_right_arm_angles=save_right_arm_angles, 
                save_left_hand_angles=save_left_hand_angles,
                save_timestamp=save_timestamp,
                save_depth=save_depth
            )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
        
    if save_landmarks:
        split_train_test_val(LANDMARK_CSV_PATH)
    if save_images and num_img_save == 0:
        shutil.rmtree(DATA_DIR)

    