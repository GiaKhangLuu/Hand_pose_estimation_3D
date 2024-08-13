"""
Convention:
    rs = opposite = REALSENSE camera
    oak = rightside = OAK camera
"""
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
MMPOSE_DIR = os.path.join(ROOT_DIR, 'mmpose')

sys.path.append(os.path.join(CURRENT_DIR, '..'))
sys.path.insert(0, MMPOSE_DIR)

from mmcv.image import imread
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import init_model as init_pose_estimator
from mmpose.apis import inference_topdown
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from mmpose.evaluation.functional import nms
from mmpose.utils import adapt_mmdet_pipeline

import yaml
import cv2
import numpy as np
import queue
import threading
import time
from functools import partial
from filterpy.kalman import KalmanFilter

from stream_3d import visualize_arm, visualize_hand
from angle_calculation import get_angles_between_joints
from camera_tools import initialize_oak_cam, initialize_realsense_cam, stream_rs, stream_oak
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

# -------------------- INIT DETECTION MODELS -------------------- 
det_config = os.path.join(MMPOSE_DIR,
    'projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py')
det_ckpt = "/home/giakhang/Downloads/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth"
pose_config = os.path.join(MMPOSE_DIR, 
    "projects/rtmpose/rtmpose/wholebody_2d_keypoint/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py")
pose_ckpt = "/home/giakhang/Downloads/rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192-c8b76419_20230728.pth"
device = 'cuda'

rs_detector = init_detector(det_config, det_ckpt, device=device)
rs_detector.cfg = adapt_mmdet_pipeline(rs_detector.cfg)
oak_detector = init_detector(det_config, det_ckpt, device=device)
oak_detector.cfg = adapt_mmdet_pipeline(oak_detector.cfg)

rs_estimator = init_pose_estimator(pose_config, pose_ckpt, device=device) # Each camera is responsible for one detector because the detector stores the pose in prev. frame
oak_estimator = init_pose_estimator(pose_config, pose_ckpt, device=device) # Each camera is responsible for one detector because the detector stores the pose in prev. frame
rs_estimator.cfg.visualizer.line_width = 2
oak_estimator.cfg.visualizer.line_width = 2

# build the visualizer
rs_visualizer = VISUALIZERS.build(rs_estimator.cfg.visualizer)
oak_visualizer = VISUALIZERS.build(oak_estimator.cfg.visualizer)

# set skeleton, colormap and joint connection rule
rs_visualizer.set_dataset_meta(rs_estimator.dataset_meta)
oak_visualizer.set_dataset_meta(oak_estimator.dataset_meta)

# -------------------- CREATE QUEUE TO INTERACT WITH THREADS -------------------- 
opposite_frame_getter_queue = queue.Queue()  # This queue directly gets RAW frame (rgb, depth) from REALSENSE camera
rightside_frame_getter_queue = queue.Queue()  # This queue directly gets RAW frame (rgb, depth) from OAK camera

arm_points_vis_queue = queue.Queue()  # This queue stores fused arm landmarks to visualize by open3d
hand_points_vis_queue = queue.Queue()  # This queue stores fused hand landmarks to visualize by open3d 

hand_to_fuse = "Left" if hand_fuse_left else "Right"
arm_to_get = "left" if arm_fuse_left else "right"
#landmarks_name_want_to_get = get_landmarks_name_based_on_arm(arm_to_get)
#landmarks_id_want_to_get = [arm_pose_landmarks_name.index(name) for name in landmarks_name_want_to_get]

#arm_hand_fused_names = landmarks_name_want_to_get.copy()
#arm_hand_fused_names.extend(hand_landmarks_name) 

if __name__ == "__main__":
    pipeline_left_oak = initialize_oak_cam(left_oak_stereo_size)
    pipeline_right_oak = initialize_oak_cam(right_oak_stereo_size)

    left_oak_thread = threading.Thread(target=stream_oak, args=(pipeline_left_oak,  
        opposite_frame_getter_queue, left_oak_mxid), daemon=True)
    right_oak_thread = threading.Thread(target=stream_oak, args=(pipeline_right_oak,  
        rightside_frame_getter_queue, right_oak_mxid), daemon=True)

    #vis_arm_thread = threading.Thread(target=visualize_arm,
        #args=(arm_points_vis_queue, 
            #arm_hand_fused_names,
            #debug_angle_j1,
            #debug_angle_j2,
            #debug_angle_j3,
            #debug_angle_j4,
            #debug_angle_j5,
            #debug_angle_j6,
            #draw_xyz,
            #True,
            #joint_vector_color,
            #ref_vector_color,),
        #daemon=True)

    left_oak_thread.start()
    right_oak_thread.start()
    #if plot_3d:
        #vis_arm_thread.start()

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

        opposite_rgb = opposite_frame_getter_queue.get()
        rightside_rgb = rightside_frame_getter_queue.get()

        opposite_rgb = cv2.resize(opposite_rgb, frame_size)
        rightside_rgb = cv2.resize(rightside_rgb, frame_size)

        #opposite_det_result = inference_detector(rs_detector, opposite_rgb)
        #opposite_pred_instance = opposite_det_result.pred_instances.cpu().numpy()
        #opposite_bboxes = np.concatenate(
            #(opposite_pred_instance.bboxes, opposite_pred_instance.scores[:, None]), axis=1)
        #opposite_bboxes = opposite_bboxes[np.logical_and(opposite_pred_instance.labels == 0,
                                    #opposite_pred_instance.scores > 0.8)]
        #opposite_bboxes = opposite_bboxes[nms(opposite_bboxes, 0.3), :4]

        #rightside_det_result = inference_detector(oak_detector, rightside_rgb)
        #rightside_pred_instance = rightside_det_result.pred_instances.cpu().numpy()
        #rightside_bboxes = np.concatenate(
            #(rightside_pred_instance.bboxes, rightside_pred_instance.scores[:, None]), axis=1)
        #rightside_bboxes = rightside_bboxes[np.logical_and(rightside_pred_instance.labels == 0,
                                    #rightside_pred_instance.scores > 0.8)]
        #rightside_bboxes = rightside_bboxes[nms(rightside_bboxes, 0.3), :4]

        opposite_detection_result = inference_topdown(rs_estimator, opposite_rgb)
        rightside_detection_result = inference_topdown(oak_estimator, rightside_rgb)

        opposite_detection_result = merge_data_samples(opposite_detection_result)
        rightside_detection_result = merge_data_samples(rightside_detection_result)

        opposite_rgb = rs_visualizer.add_datasample(
            'opposite_result',
            opposite_rgb,
            data_sample=opposite_detection_result,
            draw_gt=False,
            draw_heatmap=False,
            draw_bbox=False,
            show_kpt_idx=False,
            skeleton_style="mmpose",
            show=False,
            wait_time=0,
            kpt_thr=0.3
        )

        rightside_rgb = oak_visualizer.add_datasample(
            'rightside_result',
            rightside_rgb,
            data_sample=rightside_detection_result,
            draw_gt=False,
            draw_heatmap=False,
            draw_bbox=False,
            show_kpt_idx=False,
            skeleton_style="mmpose",
            show=False,
            wait_time=0,
            kpt_thr=0.3
        )

        opposite_landmarks = opposite_detection_result.get("pred_instances", None)
        if opposite_landmarks is not None:
            print("-------")
            print(opposite_landmarks.keypoints.shape)

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

        cv2.imshow("Frame", frame_of_two_cam)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break