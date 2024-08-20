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
from numpy.typing import NDArray
from filterpy.kalman import KalmanFilter

from stream_3d import visualize_arm, visualize_hand
from angle_calculation import get_angles_between_joints
from camera_tools import initialize_oak_cam, initialize_realsense_cam, stream_rs, stream_oak
from utilities import fuse_landmarks_from_two_cameras, convert_to_shoulder_coord
from common import (load_data_from_npz_file,
    get_oak_2_rs_matrix,
    load_config,
    get_xyZ)

# ------------------- READ CONFIG ------------------- 
config_file_path = os.path.join(CURRENT_DIR, "configuration.yaml") 
config = load_config(config_file_path)

left_oak_mxid = config["camera"]["left_camera"]["mxid"]
left_cam_calib_path = config["camera"]["left_camera"]["left_camera_calibration_path"]

right_oak_mxid = config["camera"]["right_camera"]["mxid"]
right_cam_calib_path = config["camera"]["right_camera"]["right_camera_calibration_path"]

frame_size = (config["process_frame"]["frame_size"]["width"], 
    config["process_frame"]["frame_size"]["height"])
frame_color_format = config["process_frame"]["frame_color_format"]

plot_3d = config["utilities"]["plot_3d"]
is_draw_landmarks = config["utilities"]["draw_landmarks"]
save_landmarks = config["utilities"]["save_landmarks"]

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

original_size = (720, 1280)
right_oak_ins = _scale_intrinsic_by_res(right_oak_ins, original_size, frame_size[::-1])
left_oak_ins = _scale_intrinsic_by_res(left_oak_ins, original_size, frame_size[::-1])

right_2_left_mat_avg = get_oak_2_rs_matrix(right_oak_r_raw, right_oak_t_raw, 
    left_oak_r_raw, left_oak_t_raw)

# -------------------- INIT DETECTION MODELS -------------------- 

left_detector = init_detector(
    os.path.join(MMPOSE_DIR, "projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py"), 
    "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth", 
    device="cuda")
left_detector.cfg = adapt_mmdet_pipeline(left_detector.cfg)

right_detector = init_detector(
    os.path.join(MMPOSE_DIR, "projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py"), 
    "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth", 
    device="cuda")
right_detector.cfg = adapt_mmdet_pipeline(right_detector.cfg)


#pose_config = os.path.join(MMPOSE_DIR, 
    #"configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py")
#pose_ckpt = "/home/giakhang/Downloads/rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192-c8b76419_20230728.pth"
#left_pose_config = "/home/giakhang/dev/pose_sandbox/rtmpose-m_4_left_cam.py"
#right_pose_config = "/home/giakhang/dev/pose_sandbox/rtmpose-m_4_right_cam.py"
#left_pose_ckpt = "/home/giakhang/dev/pose_sandbox/left_cam_rtmpose-m.pth"
#right_pose_ckpt = "/home/giakhang/dev/pose_sandbox/right_cam_rtmpose-m.pth"
left_pose_config = right_pose_config = os.path.join(MMPOSE_DIR,
    "configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py")
left_pose_ckpt = right_pose_ckpt = "/home/giakhang/Downloads/rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192-c8b76419_20230728.pth"
device = 'cuda'

left_cam_pose_estimator = init_pose_estimator(left_pose_config, left_pose_ckpt, device=device) # Each camera is responsible for one detector because the detector stores the pose in prev. frame
right_cam_pose_estimator = init_pose_estimator(right_pose_config, right_pose_ckpt, device=device) # Each camera is responsible for one detector because the detector stores the pose in prev. frame
left_cam_pose_estimator.cfg.visualizer.line_width = 2
right_cam_pose_estimator.cfg.visualizer.line_width = 2

# build the visualizer
left_cam_visualizer = VISUALIZERS.build(left_cam_pose_estimator.cfg.visualizer)
right_cam_visualizer = VISUALIZERS.build(right_cam_pose_estimator.cfg.visualizer)

# set skeleton, colormap and joint connection rule
left_cam_visualizer.set_dataset_meta(left_cam_pose_estimator.dataset_meta)
right_cam_visualizer.set_dataset_meta(right_cam_pose_estimator.dataset_meta)

# -------------------- CREATE QUEUE TO INTERACT WITH THREADS -------------------- 
left_frame_getter_queue = queue.Queue()  # This queue directly gets RAW frame (rgb, depth) from REALSENSE camera
right_frame_getter_queue = queue.Queue()  # This queue directly gets RAW frame (rgb, depth) from OAK camera

arm_points_vis_queue = queue.Queue()  # This queue stores fused arm landmarks to visualize by open3d

if __name__ == "__main__":
    pipeline_left_oak = initialize_oak_cam()
    pipeline_right_oak = initialize_oak_cam()

    left_oak_thread = threading.Thread(target=stream_oak, args=(pipeline_left_oak,  
        left_frame_getter_queue, left_oak_mxid), daemon=True)
    right_oak_thread = threading.Thread(target=stream_oak, args=(pipeline_right_oak,  
        right_frame_getter_queue, right_oak_mxid), daemon=True)


    landmark_dictionary = ["left shoulder", "left elbow", "left hip", "right shoulder", "right hip", 
        "WRIST", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip", "INDEX_FINGER_MCP", 
        "index_finger_pip", "index_finger_dip", "index_finger_tip", "MIDDLE_FINGER_MCP", 
        "middle_finger_pip", "middle_finger_dip", "middle_finger_tip", "ring_finger_mcp", 
        "ring_finger_pip", "ring_finger_dip", "ring_finger_tip", "pinky_mcp", "pinky_pip", 
        "pinky_dip", "pinky_tip"]

    vis_arm_thread = threading.Thread(target=visualize_arm,
        args=(arm_points_vis_queue, 
            landmark_dictionary,
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

        left_rgb, left_depth = left_frame_getter_queue.get()
        right_rgb, right_depth = right_frame_getter_queue.get()

        left_rgb = cv2.resize(left_rgb, frame_size)
        right_rgb = cv2.resize(right_rgb, frame_size)
        left_depth = cv2.resize(left_depth, frame_size)
        right_depth = cv2.resize(right_depth, frame_size)

        # predict bbox
        left_det_result = inference_detector(left_detector, left_rgb)
        left_pred_instance = left_det_result.pred_instances.cpu().numpy()
        left_bboxes = np.concatenate(
            (left_pred_instance.bboxes, left_pred_instance.scores[:, None]), axis=1)
        left_bboxes = left_bboxes[np.logical_and(left_pred_instance.labels == 0,
                                    left_pred_instance.scores > 0.8)]
        left_bboxes = left_bboxes[nms(left_bboxes, 0.5), :4]

        right_det_result = inference_detector(right_detector, right_rgb)
        right_pred_instance = right_det_result.pred_instances.cpu().numpy()
        right_bboxes = np.concatenate(
            (right_pred_instance.bboxes, right_pred_instance.scores[:, None]), axis=1)
        right_bboxes = right_bboxes[np.logical_and(right_pred_instance.labels == 0,
                                    right_pred_instance.scores > 0.8)]
        right_bboxes = right_bboxes[nms(right_bboxes, 0.5), :4]

        left_detection_result = inference_topdown(left_cam_pose_estimator, left_rgb, left_bboxes)
        right_detection_result = inference_topdown(right_cam_pose_estimator, right_rgb, right_bboxes)

        left_detection_result = merge_data_samples(left_detection_result)
        right_detection_result = merge_data_samples(right_detection_result)

        left_rgb = left_cam_visualizer.add_datasample(
            'opposite_result',
            left_rgb,
            data_sample=left_detection_result,
            draw_gt=False,
            draw_heatmap=False,
            draw_bbox=True,
            show_kpt_idx=False,
            skeleton_style="mmpose",
            show=False,
            wait_time=0,
            kpt_thr=0.3
        )

        right_rgb = right_cam_visualizer.add_datasample(
            'rightside_result',
            right_rgb,
            data_sample=right_detection_result,
            draw_gt=False,
            draw_heatmap=False,
            draw_bbox=True,
            show_kpt_idx=False,
            skeleton_style="mmpose",
            show=False,
            wait_time=0,
            kpt_thr=0.3
        )

        left_predictions = left_detection_result.get("pred_instances", None)
        right_predictions = right_detection_result.get("pred_instances", None)

        if left_predictions is not None and right_predictions is not None:
            left_body_landmarks = left_predictions.keypoints[0]
            right_body_landmarks = right_predictions.keypoints[0]

            landmarks_id_want_to_get = [5, 7, 11, 6, 12, 91, 92, 93, 94, 95, 
                96, 97, 98, 99, 100, 101, 102, 103, 104, 
                105, 106, 107, 108, 109, 110, 111]

            left_selected_landmarks = left_body_landmarks[landmarks_id_want_to_get]
            right_selected_landmarks = right_body_landmarks[landmarks_id_want_to_get]

            left_Z = get_depth(left_selected_landmarks, left_depth, 9)
            left_selected_landmarks_xyZ = np.concatenate([left_selected_landmarks, left_Z[:, None]], axis=-1)
            right_Z = get_depth(right_selected_landmarks, right_depth, 9)
            right_selected_landmarks_xyZ = np.concatenate([right_selected_landmarks, right_Z[:, None]], axis=-1)

            fused_XYZ = fuse_landmarks_from_two_cameras(left_selected_landmarks_xyZ,
                right_selected_landmarks_xyZ,
                right_oak_ins,
                left_oak_ins,
                right_2_left_mat_avg) 

            arm_hand_XYZ_wrt_shoulder, xyz_origin = convert_to_shoulder_coord(fused_XYZ, landmark_dictionary)

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

        frame_of_two_cam = np.vstack([left_rgb, right_rgb])
        frame_of_two_cam = cv2.resize(frame_of_two_cam, (640, 720))
        cv2.putText(frame_of_two_cam, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("Frame", frame_of_two_cam)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
