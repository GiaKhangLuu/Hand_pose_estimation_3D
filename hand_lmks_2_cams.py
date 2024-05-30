import os
import yaml
import cv2
import mediapipe as mp
import numpy as np
from numpy.typing import NDArray
import math
import queue
import threading
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import euclidean
from scipy.optimize import minimize
from functools import partial
from typing import Tuple
import time

from tools import (detect_hand_landmarks, 
                   filter_depth, get_xyZ, 
                   fuse_landmarks_from_two_cameras, 
                   convert_to_wrist_coord,
                   calculate_angles_between_joints,
                   plot_3d)
from camera_tools import initialize_oak_cam, initialize_realsense_cam, stream_rs, stream_oak

finger_joints_names = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

sliding_window_size = 12 # Diameter of each pixel neighborhood
sigma_color = 25 # Filter sigma in the color space for bilateral
sigma_space = 25 # Filter sigma in the coordinate space for bilateral
frame_size = (640, 360)

opposite_landmarks_detector = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
right_side_landmarks_detector = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)

opposite_frame_queue = queue.Queue()
right_side_frame_queue = queue.Queue()

opposite_depth_queue = queue.Queue()
right_side_depth_queue = queue.Queue()

opposite_detection_results_queue = queue.Queue()
right_side_detection_results_queue = queue.Queue()

opposite_landmarks_queue = queue.Queue()
right_side_landmarks_queue = queue.Queue()

# -------------------- GET TRANSFORMATION MATRIX -------------------- 
oak_data = np.load('./camera_calibration/oak_calibration.npz')
rs_data = np.load('./camera_calibration/rs_calibration.npz')

oak_r_raw, oak_t_raw, oak_ins = oak_data['rvecs'], oak_data['tvecs'], oak_data['camMatrix']
rs_r_raw, rs_t_raw, rs_ins = rs_data['rvecs'], rs_data['tvecs'], rs_data['camMatrix']

rs_r_raw = rs_r_raw.squeeze()
rs_t_raw = rs_t_raw.squeeze()
oak_r_raw = oak_r_raw.squeeze()
oak_t_raw = oak_t_raw.squeeze()

rs_r_mat = R.from_rotvec(rs_r_raw, degrees=False)
rs_r_mat = rs_r_mat.as_matrix()

oak_r_mat = R.from_rotvec(oak_r_raw, degrees=False)
oak_r_mat = oak_r_mat.as_matrix()

oak_r_t_mat = np.dstack([oak_r_mat, oak_t_raw[:, :, None]])
rs_r_t_mat = np.dstack([rs_r_mat, rs_t_raw[:, :, None]])

extra_row = np.array([0, 0, 0, 1] * oak_r_t_mat.shape[0]).reshape(-1, 4)[:, None, :]
oak_r_t_mat = np.concatenate([oak_r_t_mat, extra_row], axis=1)
rs_r_t_mat = np.concatenate([rs_r_t_mat, extra_row], axis=1)

oak_r_t_mat_inv = np.linalg.inv(oak_r_t_mat)
oak_2_rs_mat = np.matmul(rs_r_t_mat, oak_r_t_mat_inv)
oak_2_rs_mat_avg = np.average(oak_2_rs_mat, axis=0)

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
              mp_hands):

    sliding_window_size = 12 # Diameter of each pixel neighborhood
    sigma_color = 25 # Filter sigma in the color space for bilateral
    sigma_space = 25 # Filter sigma in the coordinate space for bilateral
    frame_size = (640, 360)

    while True:
        if opposite_frame_queue.empty(): 
            continue
        if right_side_frame_queue.empty():
            continue

        opposite_rgb, opposite_depth = opposite_frame_queue.get()
        right_side_rgb, right_side_depth  = right_side_frame_queue.get()

        opposite_depth = filter_depth(opposite_depth, sliding_window_size, sigma_color, sigma_space)
        assert opposite_rgb.shape[:-1] == opposite_depth.shape

        # right_side camera, so that we need to resize rgb_img
        right_side_rgb = cv2.resize(right_side_rgb, frame_size)
        right_side_depth = filter_depth(right_side_depth, sliding_window_size, sigma_color, sigma_space)
        assert right_side_rgb.shape[:-1] == right_side_depth.shape

        opposite_rgb = detect_hand_landmarks(opposite_rgb, 
                                             opposite_landmarks_detector, 
                                             opposite_landmarks_queue,
                                             mp_drawing,
                                             mp_hands)

        opposite_detection_results_queue.put(opposite_rgb)
        opposite_depth_queue.put(opposite_depth)
        if opposite_detection_results_queue.qsize() > 1:
            opposite_detection_results_queue.get()
        if opposite_depth_queue.qsize() > 1:
            opposite_depth_queue.get()

        right_side_rgb =  detect_hand_landmarks(right_side_rgb, 
                                                right_side_landmarks_detector,
                                                right_side_landmarks_queue,
                                                mp_drawing,
                                                mp_hands)

        right_side_detection_results_queue.put(right_side_rgb)
        right_side_depth_queue.put(right_side_depth)
        if right_side_detection_results_queue.qsize() > 1:
            right_side_detection_results_queue.get()
        if right_side_depth_queue.qsize() > 1:
            right_side_depth_queue.get()
        
        time.sleep(0.001)

if __name__ == "__main__":
    pipeline_rs, rsalign = initialize_realsense_cam(frame_size)
    pipeline_oak = initialize_oak_cam(frame_size)
    
    #stream_rs(pipeline_rs, rsalign, opposite_landmarks_queue)
    #stream_oak(rgb_queue_oak, depth_queue_oak, right_side_landmarks_queue)

    # Start RealSense and OAK-D processing threads
    rs_thread = threading.Thread(target=stream_rs, args=(pipeline_rs, rsalign, 
                                                         opposite_frame_queue,), daemon=True)
    oak_thread = threading.Thread(target=stream_oak, args=(pipeline_oak, frame_size, 
                                                           right_side_frame_queue,), daemon=True)
    detect_thread = threading.Thread(target=get_frame, args=(opposite_frame_queue,
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
                                                             mp_hands,), daemon=True)  

    rs_thread.start()
    oak_thread.start()
    detect_thread.start()

    while True:

        if not opposite_detection_results_queue.empty() and not right_side_detection_results_queue.empty():
            opposite_frame_result = opposite_detection_results_queue.get()
            right_side_frame_result = right_side_detection_results_queue.get()

            frame_of_two_cam = np.vstack([opposite_frame_result, right_side_frame_result])

            opposite_depth = None if opposite_depth_queue.empty() else opposite_depth_queue.get()
            right_side_depth = None if right_side_depth_queue.empty() else right_side_depth_queue.get()

            # 1. Get xyZ
            if opposite_landmarks_queue.empty() or right_side_landmarks_queue.empty():
                cv2.imshow("Frame", frame_of_two_cam)
                continue

            opposite_landmarks = opposite_landmarks_queue.get()        
            right_side_landmarks = right_side_landmarks_queue.get()        
            opposite_xyZ = get_xyZ(opposite_landmarks, opposite_depth, frame_size, sliding_window_size)
            right_side_xyZ = get_xyZ(right_side_landmarks, right_side_depth, frame_size, sliding_window_size)

            if np.isnan(opposite_xyZ).any() or np.isnan(right_side_xyZ).any():
                cv2.imshow("Frame", frame_of_two_cam)
                continue

            # 2. Fuse landmarks of two cameras
            fused_XYZ = fuse_landmarks_from_two_cameras(opposite_xyZ, right_side_xyZ,
                                                        oak_ins,
                                                        rs_ins,
                                                        oak_2_rs_mat_avg) 

            # 3. Convert to wrist coord
            writs_XYZ, fingers_XYZ_wrt_wrist = convert_to_wrist_coord(fused_XYZ)

            # 4. Calculate angles
            angles = calculate_angles_between_joints(writs_XYZ, fingers_XYZ_wrt_wrist, degrees=False)

            # 5. Plot (optional)
            plot_3d(writs_XYZ, fingers_XYZ_wrt_wrist[:, :, 0], fingers_XYZ_wrt_wrist[:, :, 1], fingers_XYZ_wrt_wrist[:, :, 2])

            cv2.imshow("Frame", frame_of_two_cam)

        
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
            cv2.destroyAllWindows()

        #time.sleep(0.001)
        #opposite_depth = None if self._opposite_depth_queue.empty() else self._opposite_depth_queue.get()
        #right_side_depth = None if self._right_side_depth_queue.empty() else self._right_side_depth_queue.get()
