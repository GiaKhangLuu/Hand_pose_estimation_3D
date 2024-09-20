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
from filterpy.kalman import KalmanFilter
from datetime import datetime

from stream_3d import visualize_arm, visualize_hand
from angle_calculation import get_angles_between_joints
from angle_calculation_v2 import calculate_six_arm_angles
from camera_tools import initialize_oak_cam, initialize_realsense_cam, stream_rs, stream_oak
from csv_writer import (create_csv, 
    append_to_csv, 
    fusion_csv_columns_name, 
    split_train_test_val,
    arm_angles_name)
from utilities import (convert_to_shoulder_coord,
    convert_to_wrist_coord,
    flatten_two_camera_input)
from kalman_filter import predict_and_correct
from common import (load_data_from_npz_file,
    get_oak_2_rs_matrix,
    load_config)
from send_data_to_robot import send_udp_message
from landmarks_detectors import LandmarksDetectors
from landmarks_fuser import LandmarksFuser

# ------------------- READ CONFIG ------------------- 
config_file_path = os.path.join(CURRENT_DIR, "configuration.yaml") 
config = load_config(config_file_path)

left_oak_mxid = config["camera"]["left_camera"]["mxid"]
left_cam_calib_path = config["camera"]["left_camera"]["left_camera_calibration_path"]
right_oak_mxid = config["camera"]["right_camera"]["mxid"]
right_cam_calib_path = config["camera"]["right_camera"]["right_camera_calibration_path"]

frame_size = (config["process_frame"]["frame_size"]["width"], 
    config["process_frame"]["frame_size"]["height"])
frame_calibrated_size = (config["camera"]["frame_calibrated_size"]["height"],
    config["camera"]["frame_calibrated_size"]["width"])

limit_angles = config["angle_limitation"]["enable"]
send_udp = config["send_udp"]
plot_3d = config["utilities"]["plot_3d"]
draw_landmarks = config["utilities"]["draw_landmarks"]
save_landmarks = config["utilities"]["save_landmarks"]
save_angles = config["utilities"]["save_angles"]
save_images = config["utilities"]["save_images"]

detection_model_selection_id = config["detection_phase"]["model_selection_id"]
detection_model_list = tuple(config["detection_phase"]["model_list"])
arm_hand_fused_names = config["detection_phase"]["fusing_landmark_dictionary"]

fusing_enable = config["fusing_phase"]["enable"]
fusing_method_list = tuple(config["fusing_phase"]["fusing_methods"])
fusing_method_selection_id = config["fusing_phase"]["fusing_selection_id"]

draw_xyz = config["debugging_mode"]["draw_xyz"]
debug_angle_j1 = config["debugging_mode"]["show_left_arm_angle_j1"]
debug_angle_j2 = config["debugging_mode"]["show_left_arm_angle_j2"]
debug_angle_j3 = config["debugging_mode"]["show_left_arm_angle_j3"]
debug_angle_j4 = config["debugging_mode"]["show_left_arm_angle_j4"]
debug_angle_j5 = config["debugging_mode"]["show_left_arm_angle_j5"]
debug_angle_j6 = config["debugging_mode"]["show_left_arm_angle_j6"]
ref_vector_color = list(config["debugging_mode"]["ref_vector_color"])
joint_vector_color = list(config["debugging_mode"]["joint_vector_color"])

joint1_min = config["angle_limitation"]["angle_joint1"]["min"]
joint1_max = config["angle_limitation"]["angle_joint1"]["max"]
joint2_min = config["angle_limitation"]["angle_joint2"]["min"]
joint2_max = config["angle_limitation"]["angle_joint2"]["max"]
joint3_min = config["angle_limitation"]["angle_joint3"]["min"]
joint3_max = config["angle_limitation"]["angle_joint3"]["max"]
joint4_min = config["angle_limitation"]["angle_joint4"]["min"]
joint4_max = config["angle_limitation"]["angle_joint4"]["max"]
joint5_min = config["angle_limitation"]["angle_joint5"]["min"]
joint5_max = config["angle_limitation"]["angle_joint5"]["max"]
joint6_min = config["angle_limitation"]["angle_joint6"]["min"]
joint6_max = config["angle_limitation"]["angle_joint6"]["max"]
angles_min = [joint1_min, joint2_min, joint3_min, joint4_min, joint5_min, joint6_min]
angles_max = [joint1_max, joint2_max, joint3_max, joint4_max, joint5_max, joint6_max]

use_fusing_network = fusing_method_list[fusing_method_selection_id] != "minimize_distance" 

client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_ip = config["socket"]["server_ip"]
server_port = config["socket"]["server_port"]

run_to_collect_data = config["run_to_collect_data_for_fusing_and_detection"]

if run_to_collect_data:
    plot_3d = True
    send_udp = False
    save_landmarks = True
    save_images = True
    #detection_model_selection_id = 0
    fusing_enable = True
    fusing_method_selection_id = 1
    use_fusing_network = False
    config["detection_phase"]["mediapipe"]["hand_detection"]["is_enable"] = True
    config["detection_phase"]["mediapipe"]["body_detection"]["is_enable"] = True
    config["fusing_phase"]["minimize_distance"]["tolerance"] = 1e-10


if (use_fusing_network):
    save_landmarks = False

# -------------------- GET TRANSFORMATION MATRIX -------------------- 
right_cam_data = load_data_from_npz_file(right_cam_calib_path)
left_cam_data = load_data_from_npz_file(left_cam_calib_path)

right_oak_r_raw, right_oak_t_raw, right_camera_intr = right_cam_data['rvecs'], right_cam_data['tvecs'], right_cam_data['camMatrix']
left_oak_r_raw, left_oak_t_raw, left_camera_intr = left_cam_data['rvecs'], left_cam_data['tvecs'], left_cam_data['camMatrix']

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

right_camera_intr = _scale_intrinsic_by_res(right_camera_intr, frame_calibrated_size, frame_size[::-1])
left_camera_intr = _scale_intrinsic_by_res(left_camera_intr, frame_calibrated_size, frame_size[::-1])

right_2_left_mat_avg = get_oak_2_rs_matrix(right_oak_r_raw, right_oak_t_raw, 
    left_oak_r_raw, left_oak_t_raw)

# -------------------- CREATE QUEUE TO INTERACT WITH THREADS -------------------- 
left_frame_getter_queue = queue.Queue()  # This queue directly gets RAW frame (rgb, depth) from REALSENSE camera
right_frame_getter_queue = queue.Queue()  # This queue directly gets RAW frame (rgb, depth) from OAK camera

arm_points_vis_queue = queue.Queue()  # This queue stores fused arm landmarks to visualize by open3d
hand_points_vis_queue = queue.Queue()  # This queue stores fused hand landmarks to visualize by open3d 
TARGET_ANGLES_QUEUE = queue.Queue()  # This queue stores target angles 

landmarks_detectors = LandmarksDetectors(detection_model_selection_id, 
    detection_model_list, config["detection_phase"], draw_landmarks)
landmarks_fuser = LandmarksFuser(fusing_method_selection_id,
    fusing_method_list, config["fusing_phase"], frame_size)

if __name__ == "__main__":
    if (save_landmarks or
        save_angles or
        save_images):
        current_time = datetime.now().strftime('%Y-%m-%d-%H:%M')
        current_date = datetime.now().strftime('%Y-%m-%d')
        DATA_DIR = os.path.join(CURRENT_DIR, "../..", "data", current_date)
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

        if save_angles:
            ARM_ANGLE_CSV_PATH = os.path.join(DATA_DIR, "arm_angle_{}.csv".format(current_time))
            create_csv(ARM_ANGLE_CSV_PATH, ["frame", "angle_j1"])

    pipeline_left_oak = initialize_oak_cam()
    pipeline_right_oak = initialize_oak_cam()

    left_oak_thread = threading.Thread(target=stream_oak, args=(pipeline_left_oak,  
        left_frame_getter_queue, left_oak_mxid), daemon=True)
    right_oak_thread = threading.Thread(target=stream_oak, args=(pipeline_right_oak,  
        right_frame_getter_queue, right_oak_mxid), daemon=True)

    left_oak_thread.start()
    right_oak_thread.start()

    if send_udp:
        send_data_thread = threading.Thread(target=send_udp_message, args=(TARGET_ANGLES_QUEUE,
            True, 0.5,), daemon=True)
        send_data_thread.start()

    if plot_3d:
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
        vis_arm_thread.start()
        #vis_hand_thread.start()

    timestamp = 0
    frame_count = 0
    num_img_save = 0
    start_time = time.time()
    fps = 0

    while True:
        left_camera_body_landmarks_xyZ, right_camera_body_landmarks_xyZ = None, None
        # Get and preprocess rgb and its depth
        left_rgb, left_depth = left_frame_getter_queue.get()
        right_rgb, right_depth = right_frame_getter_queue.get()

        left_rgb = cv2.resize(left_rgb, frame_size)
        right_rgb = cv2.resize(right_rgb, frame_size)
        left_depth = cv2.resize(left_depth, frame_size)
        right_depth = cv2.resize(right_depth, frame_size)

        left_image_to_save = np.copy(left_rgb)
        right_image_to_save = np.copy(right_rgb)

        # Detection phase
        left_det_rs = landmarks_detectors.detect(left_rgb, left_depth, timestamp, side="left")
        right_det_rs = landmarks_detectors.detect(right_rgb, right_depth, timestamp, side="right")

        left_camera_body_landmarks_xyZ = left_det_rs["detection_result"]
        right_camera_body_landmarks_xyZ = right_det_rs["detection_result"]
        if draw_landmarks:
            left_rgb = left_det_rs["drawed_img"]
            right_rgb = right_det_rs["drawed_img"]

        # Fusing phase
        if (fusing_enable and
            left_camera_body_landmarks_xyZ is not None and 
            right_camera_body_landmarks_xyZ is not None and
            left_camera_body_landmarks_xyZ.shape[0] == len(arm_hand_fused_names) and
            left_camera_body_landmarks_xyZ.shape[0] == right_camera_body_landmarks_xyZ.shape[0]):
            # JUST FUSE TO XYZ, MANUAL CONVERT TO SHOULDER_COORDINATE
            arm_hand_fused_XYZ = landmarks_fuser.fuse(left_camera_body_landmarks_xyZ,
                right_camera_body_landmarks_xyZ,
                left_camera_intr,
                right_camera_intr,
                right_2_left_mat_avg)

            arm_hand_XYZ_wrt_shoulder, xyz_origin = convert_to_shoulder_coord(
                arm_hand_fused_XYZ,
                arm_hand_fused_names)

            if (arm_hand_XYZ_wrt_shoulder is not None and
                xyz_origin is not None):

                angles, rot_mats_wrt_origin, rot_mats_wrt_parent = calculate_six_arm_angles(arm_hand_XYZ_wrt_shoulder,
                    xyz_origin,
                    arm_hand_fused_names)
                angle_j1, angle_j2, angle_j3, angle_j4, _, _ = angles
                j1_rm, j2_rm, j3_rm, j4_rm, _, _ = rot_mats_wrt_origin
                j1_rm_wrt_parent, j2_rm_wrt_parent, j3_rm_wrt_parent, j4_rm_wrt_parent, _, _ = rot_mats_wrt_parent

                #arm_angles = get_angles_between_joints(arm_hand_XYZ_wrt_shoulder, 
                    #arm_hand_fused_names, xyz_origin)
                #angle_j1, j1_rot_mat_wrt_world = calculate_angle_j1(arm_hand_XYZ_wrt_shoulder,
                    #arm_hand_fused_names)
                #angle_j2, j2_rot_mat_wrt_world, j2_rot_mat_wrt_j1 = calculate_angle_j2(arm_hand_XYZ_wrt_shoulder,
                    #arm_hand_fused_names, j1_rot_mat_wrt_world, angle_j1)
                arm_angles = np.array([angle_j1, angle_j2, angle_j3, angle_j4, 0, 0])
                if send_udp:
                    TARGET_ANGLES_QUEUE.put((arm_angles, timestamp))
                    if TARGET_ANGLES_QUEUE.qsize() > 1:
                        TARGET_ANGLES_QUEUE.get()

                if plot_3d: 
                    arm_points_vis_queue.put((arm_hand_XYZ_wrt_shoulder, xyz_origin))
                    if arm_points_vis_queue.qsize() > 1:
                        arm_points_vis_queue.get()

                if save_angles:
                    #angles_writed_to_file = [timestamp] 
                    #angles_writed_to_file.extend(arm_angles.tolist())
                    #angles_writed_to_file.extend(filter_angles.tolist())
                    #angles_writed_to_file.append(elapsed_sending_data_time)
                    angles_writed_to_file = [timestamp, angle_j1, angle_j2]
                    append_to_csv(ARM_ANGLE_CSV_PATH, angles_writed_to_file)

                if save_landmarks and timestamp > 100:  # warm up for 100 frames before saving landmarks
                    output_landmarks = np.concatenate([arm_hand_fused_XYZ, 
                        np.zeros((48 - arm_hand_fused_XYZ.shape[0], 3))])
                    input_row = flatten_two_camera_input(left_camera_body_landmarks_xyZ,
                        right_camera_body_landmarks_xyZ,
                        left_camera_intr,
                        right_camera_intr,
                        right_2_left_mat_avg,
                        frame_size,
                        mode="ground_truth",
                        timestamp=timestamp,
                        output_landmarks=output_landmarks)
                    append_to_csv(LANDMARK_CSV_PATH, input_row)
            
                if save_images and timestamp > 100:  # warm up for 100 frames before saving image 
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

        timestamp += 1

        frame_of_two_cam = np.vstack([left_rgb, right_rgb])
        frame_of_two_cam = cv2.resize(frame_of_two_cam, (640, 720))
        cv2.putText(frame_of_two_cam, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        #rightside_rgb = cv2.resize(rightside_rgb, (960, 540))
        #cv2.putText(rightside_rgb, f'FPS: {fps:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4, cv2.LINE_AA)

        cv2.imshow("Frame", frame_of_two_cam)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
    if save_landmarks:
        split_train_test_val(LANDMARK_CSV_PATH)
    if save_images and num_img_save == 0:
        shutil.rmtree(DATA_DIR)

    