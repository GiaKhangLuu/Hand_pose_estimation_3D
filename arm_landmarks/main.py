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

from camera_tools import initialize_oak_cam, initialize_realsense_cam, stream_rs, stream_oak
from arm_landmarks.stream_w_open3d import visualization_thread
from arm_landmarks.tools import (detect_arm_landmarks, 
                                 load_config, 
                                 filter_depth,
                                 get_xyZ,
                                 load_data_from_npz_file,
                                 get_oak_2_rs_matrix,
                                 fuse_landmarks_from_two_cameras,
                                 convert_to_shoulder_coord,
                                 get_landmarks_name_based_on_arm)
from arm_landmarks.angle_calculation import get_angles_between_joints

config_file_path = os.path.join(CURRENT_DIR, "configurations/arm_landmarks.yaml") 
config = load_config(config_file_path)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

frame_size = (config["frame_width"], config["frame_height"])
sliding_window_size = config["sliding_window_size"]
sigma_color =  config["sigma_color"]
sigma_space = config["sigma_space"]       

rightside_cam_calib_path = config["camera"]["rightside_camera_calibration_path"]
opposite_cam_calib_path = config["camera"]["opposite_camera_calibration_path"]

mediapipe_min_det_conf = config["mediapipe"]["min_detection_confidence"]
mediapipe_min_tracking_conf = config["mediapipe"]["min_tracking_confidence"]
mediapipe_model_complexity = config["mediapipe"]["model_complexity"]
is_activated = config["mediapipe"]["is_activated"]
visibility_threshold = config["mediapipe"]["visibility_threshold"]

pose_landmark_names = tuple(config["pose_landmarks"])

arm_to_get = config["arm_to_get"]

plot_3d = config["utilities"]["plot_3d"]

opposite_arm_detector = mp_pose.Pose(min_detection_confidence=mediapipe_min_det_conf, 
                                     min_tracking_confidence=mediapipe_min_tracking_conf,
                                     model_complexity=mediapipe_model_complexity)
rightside_arm_detector = mp_pose.Pose(min_detection_confidence=mediapipe_min_det_conf, 
                                      min_tracking_confidence=mediapipe_min_tracking_conf,
                                      model_complexity=mediapipe_model_complexity)

opposite_frame_queue = queue.Queue()
rightside_frame_queue = queue.Queue()

opposite_depth_queue = queue.Queue()
rightside_depth_queue = queue.Queue()

opposite_arm_landmarks_queue = queue.Queue()
rightside_arm_landmarks_queue = queue.Queue()

opposite_detection_results_queue = queue.Queue()
rightside_detection_results_queue = queue.Queue()

visualization_queue = queue.Queue()

# -------------------- GET TRANSFORMATION MATRIX -------------------- 
oak_data = load_data_from_npz_file(rightside_cam_calib_path)
rs_data = load_data_from_npz_file(opposite_cam_calib_path)

oak_r_raw, oak_t_raw, oak_ins = oak_data['rvecs'], oak_data['tvecs'], oak_data['camMatrix']
rs_r_raw, rs_t_raw, rs_ins = rs_data['rvecs'], rs_data['tvecs'], rs_data['camMatrix']

oak_2_rs_mat_avg = get_oak_2_rs_matrix(oak_r_raw, oak_t_raw, 
                                       rs_r_raw, rs_t_raw)

def get_frame(opposite_frame_queue, 
              right_side_frame_queue, 
              opposite_depth_queue, 
              right_side_depth_queue,
              opposite_landmarks_detector,
              opposite_landmarks_queue,
              opposite_detection_results_queue,
              right_side_landmarks_detector,
              right_side_landmarks_queue,
              right_side_detection_results_queue,
              mp_drawing,
              mp_pose,
              is_detected):
    """
    Detect arm's landmarks from two cameras
    """

    while True:
        if opposite_frame_queue.empty(): 
            continue
        if right_side_frame_queue.empty():
            continue

        opposite_rgb, opposite_depth = opposite_frame_queue.get()
        right_side_rgb, right_side_depth  = right_side_frame_queue.get()

        opposite_depth = filter_depth(opposite_depth, sliding_window_size, sigma_color, sigma_space)
        assert opposite_rgb.shape[:-1] == opposite_depth.shape

        # right_side camera is oak, so that we need to resize rgb_img
        right_side_rgb = cv2.resize(right_side_rgb, frame_size)
        right_side_depth = filter_depth(right_side_depth, sliding_window_size, sigma_color, sigma_space)
        assert right_side_rgb.shape[:-1] == right_side_depth.shape

        if is_detected:
            opposite_rgb = detect_arm_landmarks(opposite_rgb, 
                                                opposite_landmarks_detector, 
                                                opposite_landmarks_queue,
                                                mp_drawing,
                                                mp_pose)

        opposite_detection_results_queue.put(opposite_rgb)
        opposite_depth_queue.put(opposite_depth)
        if opposite_detection_results_queue.qsize() > 1:
            opposite_detection_results_queue.get()
        if opposite_depth_queue.qsize() > 1:
            opposite_depth_queue.get()

        if is_detected:
            right_side_rgb = detect_arm_landmarks(right_side_rgb, 
                                                    right_side_landmarks_detector,
                                                    right_side_landmarks_queue,
                                                    mp_drawing,
                                                    mp_pose)

        right_side_detection_results_queue.put(right_side_rgb)
        right_side_depth_queue.put(right_side_depth)
        if right_side_detection_results_queue.qsize() > 1:
            right_side_detection_results_queue.get()
        if right_side_depth_queue.qsize() > 1:
            right_side_depth_queue.get()
        
        #time.sleep(0.001)


if __name__ == "__main__":
    pipeline_rs, rsalign = initialize_realsense_cam(frame_size)
    pipeline_oak = initialize_oak_cam(frame_size)

    rs_thread = threading.Thread(target=stream_rs, args=(pipeline_rs, rsalign, 
                                                        opposite_frame_queue,), daemon=True)

    oak_thread = threading.Thread(target=stream_oak, args=(pipeline_oak, frame_size, 
                                                        rightside_frame_queue,), daemon=True)

    detect_thread = threading.Thread(target=get_frame, args=(opposite_frame_queue,
                                                             rightside_frame_queue,
                                                             opposite_depth_queue,
                                                             rightside_depth_queue,
                                                             opposite_arm_detector,
                                                             opposite_arm_landmarks_queue,
                                                             opposite_detection_results_queue,
                                                             rightside_arm_detector,
                                                             rightside_arm_landmarks_queue,
                                                             rightside_detection_results_queue,
                                                             mp_drawing,
                                                             mp_pose,
                                                             is_activated,), daemon=True)  

    vis_thread = threading.Thread(target=visualization_thread, args=(visualization_queue,), daemon=True)

    frame_count = 0
    start_time = time.time()
    fps = 0

    rs_thread.start()
    oak_thread.start()
    detect_thread.start()
    vis_thread.start()

    count = 0

    while True:
        if not opposite_detection_results_queue.empty() and not rightside_detection_results_queue.empty():
            opposite_frame_result = opposite_detection_results_queue.get()
            right_side_frame_result = rightside_detection_results_queue.get()
            frame_of_two_cam = np.vstack([opposite_frame_result, right_side_frame_result])

            opposite_depth = None if opposite_depth_queue.empty() else opposite_depth_queue.get()
            rightside_depth = None if rightside_depth_queue.empty() else rightside_depth_queue.get()

            # 1. Get xyZ
            if (opposite_arm_landmarks_queue.empty() or 
                rightside_arm_landmarks_queue.empty() or
                opposite_depth is None or
                rightside_depth is None):
                cv2.imshow("Frame", frame_of_two_cam)
                continue

            opposite_arm_landmarks = opposite_arm_landmarks_queue.get()        
            rightside_arm_landmarks = rightside_arm_landmarks_queue.get()        

            landmarks_name_want_to_get = get_landmarks_name_based_on_arm(arm_to_get)
            landmarks_id_want_to_get = [pose_landmark_names.index(name) for name in landmarks_name_want_to_get]

            opposite_arm_xyZ = get_xyZ(opposite_arm_landmarks , 
                                       opposite_depth, 
                                       frame_size, 
                                       sliding_window_size,
                                       landmarks_id_want_to_get,
                                       visibility_threshold)  # shape: (N, 3)
            rightside_arm_xyZ = get_xyZ(rightside_arm_landmarks , 
                                        rightside_depth, 
                                        frame_size, 
                                        sliding_window_size,
                                        landmarks_id_want_to_get,
                                        visibility_threshold)  # shape: (N, 3)

            if (np.isnan(opposite_arm_xyZ).any() or 
                np.isnan(rightside_arm_xyZ).any() or 
                len(opposite_arm_xyZ) != len(rightside_arm_xyZ) or 
                len(opposite_arm_xyZ) == 0 or
                len(rightside_arm_xyZ) == 0):
                cv2.imshow("Frame", frame_of_two_cam)
                continue

            # 2. Fusing landmarks of two cameras
            fused_XYZ = fuse_landmarks_from_two_cameras(opposite_arm_xyZ, rightside_arm_xyZ,  # (N, 3)
                                                        oak_ins,
                                                        rs_ins,
                                                        oak_2_rs_mat_avg) 

            if fused_XYZ.shape[0] != len(landmarks_name_want_to_get):
                cv2.imshow("Frame", frame_of_two_cam)
                continue

            # 3. Convert to shoulder coord
            XYZ_wrt_shoulder = convert_to_shoulder_coord(fused_XYZ, landmarks_name_want_to_get)  # shapes: (3,) and ()

            # 4. Calculate angles
            angles = get_angles_between_joints(XYZ_wrt_shoulder, landmarks_name_want_to_get)

            print("-----")
            count += 1
            print("count: ", count)
            print("XYZ_wrt_shoulder: ", XYZ_wrt_shoulder)
            if count == 60:
                break

            # 5. Plot (optional)
            if plot_3d:
                visualization_queue.put(XYZ_wrt_shoulder)
                if visualization_queue.qsize() > 1:
                    visualization_queue.get()

            frame_count += 1
            # Calculate elapsed time
            elapsed_time = time.time() - start_time

            # Update FPS every second
            if elapsed_time > 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()

            # Overlay the FPS on the frame
            cv2.putText(frame_of_two_cam, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow("Frame", frame_of_two_cam)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            pipeline_rs.stop()
            break
    