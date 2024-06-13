import sys
import os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, '..'))

import yaml
import cv2
import mediapipe as mp
import numpy as np
from numpy.typing import NDArray
import math
import queue
import threading
import time
import tensorflow as tf
import torch
import joblib

from hand_landmarks.tools import (detect_hand_landmarks, 
                                  filter_depth, get_xyZ, 
                                  fuse_landmarks_from_two_cameras, 
                                  convert_to_wrist_coord,
                                  load_data_from_npz_file,
                                  get_oak_2_rs_matrix,
                                  xyZ_to_XYZ)
from camera_tools import initialize_oak_cam, initialize_realsense_cam, stream_rs, stream_oak
from hand_landmarks.stream_w_open3d import visualization_thread
from hand_landmarks.write_lmks_to_file import write_lnmks_to_file
from hand_landmarks.angle_calculation import get_angles_of_hand_joints
from hand_landmarks.neural_networks.mlp import MLP

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

wrist_and_hand_lmks_queue = queue.Queue()

write_queue = queue.Queue()

# -------------------- GET TRANSFORMATION MATRIX -------------------- 
oak_data = load_data_from_npz_file('./camera_calibration/oak_calibration.npz')
rs_data = load_data_from_npz_file('./camera_calibration/rs_calibration.npz')

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
              mp_hands):
    """
    Detect hand's landmarks from two cameras
    """

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

    use_neural_network = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    scaler_input = None
    scaler_output = None
    print("Using {} for inference: ".format(device))

    plot_3d = True  # Plot the hand landmarks on open3d
    save_landmarks = False  # Write landmarks to to file
    run_online = True  # If run_online, get landmarks from 2 cameras. Otherwise, get landmarks from file

    if use_neural_network:
        model_weight_path = "/home/giakhang/dev/Hand_pose_estimation_3D/hand_landmarks/test/best_model.pth"
        model = MLP()
        model.load_state_dict(torch.load(model_weight_path))
        model.eval()
        model.to(device)

        scaler_input = joblib.load("/home/giakhang/dev/Hand_pose_estimation_3D/hand_landmarks/test/scaler_input.pkl")
        scaler_output = joblib.load("/home/giakhang/dev/Hand_pose_estimation_3D/hand_landmarks/test/scaler_output.pkl")

    with tf.device("/GPU:0"):
        pipeline_rs, rsalign = initialize_realsense_cam(frame_size)
        pipeline_oak = initialize_oak_cam(frame_size)
                
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
        vis_thread = threading.Thread(target=visualization_thread, args=(wrist_and_hand_lmks_queue,), daemon=True)

        # UNCOMMENT THIS THREAD TO SAVE LANDMARKS
        write_lmks_thread = threading.Thread(target=write_lnmks_to_file, args=(write_queue,), daemon=True)

        # Initialize variables
        frame_count = 0
        start_time = time.time()
        fps = 0

        rs_thread.start()
        oak_thread.start()
        detect_thread.start()
        vis_thread.start()
        write_lmks_thread.start()

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
                opposite_xyZ = get_xyZ(opposite_landmarks, opposite_depth, frame_size, sliding_window_size)  # shape: (21, 3)
                right_side_xyZ = get_xyZ(right_side_landmarks, right_side_depth, frame_size, sliding_window_size)  # shape: (21, 3)

                if np.isnan(opposite_xyZ).any() or np.isnan(right_side_xyZ).any():
                    cv2.imshow("Frame", frame_of_two_cam)
                    continue

                if not use_neural_network:
                    # 2. Fuse landmarks of two cameras
                    fused_XYZ = fuse_landmarks_from_two_cameras(opposite_xyZ, right_side_xyZ,  # (21, 3)
                                                                oak_ins,
                                                                rs_ins,
                                                                oak_2_rs_mat_avg) 

                    # 3. Convert to wrist coord
                    wrist_XYZ, fingers_XYZ_wrt_wrist = convert_to_wrist_coord(fused_XYZ)  # shapes: (3,), (5, 4, 3)
                else:
                    # If using neural_network, a model predicts all landmarks w.r.t. wrist coordinates
                    # Need to convert landmarks_of_rightside_cam to opposite_cam coordinate and normalize
                    raw_XYZ_of_opposite_cam = xyZ_to_XYZ(opposite_xyZ, rs_ins)  # shape: (21, 3)
                    raw_XYZ_of_right_side_cam = xyZ_to_XYZ(right_side_xyZ, oak_ins)  # shape: (21, 3)
                    homo = np.ones(shape=raw_XYZ_of_right_side_cam.shape[0])
                    raw_XYZ_of_right_side_cam_in_opposite_cam = np.concatenate([raw_XYZ_of_right_side_cam, homo[:, None]], axis=1)  # shape: (21, 4)
                    raw_XYZ_of_right_side_cam_in_opposite_cam = np.matmul(oak_2_rs_mat_avg,   # shape: (4, 21)
                                                                          raw_XYZ_of_right_side_cam_in_opposite_cam.T)
                    raw_XYZ_of_right_side_cam_in_opposite_cam = raw_XYZ_of_right_side_cam_in_opposite_cam.T  # shape: (21, 4)
                    raw_XYZ_of_right_side_cam_in_opposite_cam = raw_XYZ_of_right_side_cam_in_opposite_cam[:, :-1]  # shape: (21, 3)

                    lmks_input = np.concatenate([raw_XYZ_of_opposite_cam, 
                                                 raw_XYZ_of_right_side_cam_in_opposite_cam], axis=0)  # shape: (42, 3)
                    lmks_input = lmks_input.flatten()[None, :]  # shape: (1, 126)
                    lmks_input = scaler_input.transform(lmks_input)
                    lmks_input = torch.tensor(lmks_input, dtype=torch.float32).to(device)

                    lmks_output = model(lmks_input)  # shape: (1, 63)
                    lmks_output = lmks_output.detach().to("cpu").numpy()
                    lmks_output = scaler_output.inverse_transform(lmks_output)
                    lmks_output = lmks_output[0]  # shape: (63,)
                    lmks_output = lmks_output.reshape(-1, 3)  # shape: (21, 3)

                    wrist_XYZ, fingers_XYZ_wrt_wrist = lmks_output[0, :], lmks_output[1:, :]
                    fingers_XYZ_wrt_wrist = fingers_XYZ_wrt_wrist.reshape(5, 4, 3) 

                # 4. Calculate angles
                angles = get_angles_of_hand_joints(wrist_XYZ, fingers_XYZ_wrt_wrist, degrees=True)

                # 5. Plot (optional)
                if plot_3d:
                    wrist_and_hand_lmks_queue.put((wrist_XYZ, fingers_XYZ_wrt_wrist))
                    if wrist_and_hand_lmks_queue.qsize() > 1:
                        wrist_and_hand_lmks_queue.get()
                
                # 6. Save landmarks (optional)
                if save_landmarks:
                    raw_XYZ_of_opposite_cam = xyZ_to_XYZ(opposite_xyZ, rs_ins)  # shape: (21, 3)
                    raw_XYZ_of_right_side_cam = xyZ_to_XYZ(right_side_xyZ, oak_ins)  # shape: (21, 3)
                    homo = np.ones(shape=raw_XYZ_of_right_side_cam.shape[0])
                    raw_XYZ_of_right_side_cam_in_opposite_cam = np.concatenate([raw_XYZ_of_right_side_cam, homo[:, None]], axis=1)  # shape: (21, 4)
                    raw_XYZ_of_right_side_cam_in_opposite_cam = np.matmul(oak_2_rs_mat_avg,   # shape: (4, 21)
                                                                        raw_XYZ_of_right_side_cam_in_opposite_cam.T)
                    raw_XYZ_of_right_side_cam_in_opposite_cam = raw_XYZ_of_right_side_cam_in_opposite_cam.T  # shape: (21, 4)

                    write_queue.put((raw_XYZ_of_opposite_cam,
                                    raw_XYZ_of_right_side_cam_in_opposite_cam[:, :-1],
                                    wrist_XYZ, 
                                    fingers_XYZ_wrt_wrist))
                    if write_queue.qsize() > 1:
                        write_queue.get()

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

            time.sleep(0.001)
