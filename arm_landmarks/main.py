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
                                 get_landmarks_name_based_on_arm,
                                 draw_landmarks_on_image,
                                 get_normalized_pose_landmarks)
from arm_landmarks.write_landmarks_to_file import write_lnmks_to_file
from arm_landmarks.angle_calculation import get_angles_between_joints

config_file_path = os.path.join(CURRENT_DIR, "configurations/arm_landmarks.yaml") 
config = load_config(config_file_path)

realsense_rgb_size = (config["realsense"]["rgb"]["width"], config["realsense"]["rgb"]["height"])
realsense_depth_size = (config["realsense"]["depth"]["width"], config["realsense"]["depth"]["height"]) 

oak_stereo_size = (config["oak"]["stereo"]["width"], config["oak"]["stereo"]["height"]) 

frame_size = (config["process_frame_size"]["width"], config["process_frame_size"]["height"])

sliding_window_size = config["sliding_window_size"]
sigma_color =  config["sigma_color"]
sigma_space = config["sigma_space"]       

rightside_cam_calib_path = config["camera"]["rightside_camera_calibration_path"]
opposite_cam_calib_path = config["camera"]["opposite_camera_calibration_path"]

mediapipe_min_det_conf = config["mediapipe"]["min_detection_confidence"]
mediapipe_min_tracking_conf = config["mediapipe"]["min_tracking_confidence"]
is_activated = config["mediapipe"]["is_activated"]
visibility_threshold = config["mediapipe"]["visibility_threshold"]
model_asset_path = config["mediapipe"]["model_asset_path"]
num_pose = config["mediapipe"]["num_pose"]

pose_landmark_names = tuple(config["pose_landmarks"])

arm_to_get = config["arm_to_get"]

plot_3d = config["utilities"]["plot_3d"]
is_draw_landmarks = config["utilities"]["draw_landmarks"]
save_landmarks = config["utilities"]["save_landmarks"]

opposite_frame_getter_queue = queue.Queue()  # This queue directly gets RAW frame (rgb, depth) from REALSENSE camera
rightside_frame_getter_queue = queue.Queue()  # This queue directly gets RAW frame (rgb, depth) from OAK camera

opposite_arm_landmarks_queue = queue.Queue()  # This queue stores arm landmarks (detection results) from REALSENSE camera
rightside_arm_landmarks_queue = queue.Queue()  # This queue stores arm landmarks (detection results) from OAK camera

opposite_streamed_frame_queue = queue.Queue()  # This queue stores processed frames (rgb, depth) from REALSENSE to stream and draw landmarks on 
rightside_streamed_frame_queue = queue.Queue()  # This queue stores processed frames (rgb, depth) from OAK to steram and draw landmarks on

visualization_queue = queue.Queue()  # This queue stores fused landmarks to visualize by open3d

write_queue = queue.Queue()  # This queue stores fused landmarks to write to file

base_options = python.BaseOptions(model_asset_path=model_asset_path,
                                  delegate=mp.tasks.BaseOptions.Delegate.GPU)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_poses=num_pose,
    output_segmentation_masks=False)

opposite_arm_detector = vision.PoseLandmarker.create_from_options(options)
rightside_arm_detector = vision.PoseLandmarker.create_from_options(options)

# -------------------- GET TRANSFORMATION MATRIX -------------------- 
oak_data = load_data_from_npz_file(rightside_cam_calib_path)
rs_data = load_data_from_npz_file(opposite_cam_calib_path)

oak_r_raw, oak_t_raw, oak_ins = oak_data['rvecs'], oak_data['tvecs'], oak_data['camMatrix']
rs_r_raw, rs_t_raw, rs_ins = rs_data['rvecs'], rs_data['tvecs'], rs_data['camMatrix']

oak_2_rs_mat_avg = get_oak_2_rs_matrix(oak_r_raw, oak_t_raw, 
                                       rs_r_raw, rs_t_raw)

def get_frame(opposite_frame_queue, 
              right_side_frame_queue, 
              opposite_streamed_frame_queue,
              rightside_streamed_frame_queue,
              opposite_landmarks_detector,
              right_side_landmarks_detector,
              opposite_landmarks_queue,
              rightside_landmarks_queue,
              is_detected):
    """
    Detect arm's landmarks from two cameras
    Input:
        opposite_frame_queue: get RAW rgb and depth from realsense camera to process
        right_side_frame_queue: get RAW rgb and depth from oak camera to process
        opposite_streamed_frame_queue: put processed rgb and depth from realsense camera to stream
        rightside_streamed_frame_queue: put processed rgb and depth from oak camera to stream
        opposite_landmarks_detector: arm landmarker for realsense camera
        right_side_landmarks_detector: arm landmarker for oak camera
        is_detected: whether to detect arm landmarks
    """

    while True:
        if opposite_frame_queue.empty(): 
            continue
        if right_side_frame_queue.empty():
            continue

        opposite_rgb, opposite_depth = opposite_frame_queue.get()
        rightside_rgb, rightside_depth = right_side_frame_queue.get()

        opposite_rgb = cv2.resize(opposite_rgb, frame_size)
        opposite_depth = cv2.resize(opposite_depth, frame_size)
        rightside_rgb = cv2.resize(rightside_rgb, frame_size)
        rightside_depth = cv2.resize(rightside_depth, frame_size)

        assert opposite_depth.shape == opposite_rgb.shape[:-1]
        assert rightside_depth.shape == rightside_rgb.shape[:-1]

        opposite_depth = filter_depth(opposite_depth, sliding_window_size, sigma_color, sigma_space)
        rightside_depth = filter_depth(rightside_depth, sliding_window_size, sigma_color, sigma_space)

        if is_detected:
            detect_arm_landmarks(opposite_rgb, 
                                 opposite_landmarks_detector, 
                                 opposite_landmarks_queue,             
                                 image_format="bgr")
            detect_arm_landmarks(rightside_rgb, 
                                 right_side_landmarks_detector,
                                 rightside_landmarks_queue,                  
                                 image_format="bgr")

        opposite_streamed_frame_queue.put((opposite_rgb,
                                           opposite_depth))
        rightside_streamed_frame_queue.put((rightside_rgb,
                                            rightside_depth))

        if opposite_streamed_frame_queue.qsize() > 1:
            opposite_streamed_frame_queue.get()

        if rightside_streamed_frame_queue.qsize() > 1:
            rightside_streamed_frame_queue.get()
        
if __name__ == "__main__":
    pipeline_rs, rsalign = initialize_realsense_cam(realsense_rgb_size, realsense_depth_size)
    pipeline_oak = initialize_oak_cam(oak_stereo_size)

    rs_thread = threading.Thread(target=stream_rs, args=(pipeline_rs, rsalign, 
                                                        opposite_frame_getter_queue,), daemon=True)

    oak_thread = threading.Thread(target=stream_oak, args=(pipeline_oak,  
                                                           rightside_frame_getter_queue,), daemon=True)

    detect_thread = threading.Thread(target=get_frame, args=(opposite_frame_getter_queue,
                                                             rightside_frame_getter_queue,
                                                             opposite_streamed_frame_queue,
                                                             rightside_streamed_frame_queue,
                                                             opposite_arm_detector,
                                                             rightside_arm_detector,
                                                             opposite_arm_landmarks_queue,
                                                             rightside_arm_landmarks_queue,
                                                             is_activated,), daemon=True)  

    vis_thread = threading.Thread(target=visualization_thread, args=(visualization_queue,), daemon=True)

    write_to_file_thread = threading.Thread(target=write_lnmks_to_file, args=(write_queue,), daemon=True)

    frame_count = 0
    start_time = time.time()
    fps = 0

    rs_thread.start()
    oak_thread.start()
    detect_thread.start()
    vis_thread.start()
    write_to_file_thread.start()

    while True:
        #if (not opposite_arm_landmarks_queue.empty() and
            #not rightside_arm_landmarks_queue.empty()):

            #opposite_result, opposite_rgb, oppo_timestamp = opposite_arm_landmarks_queue.get()
            #rightside_result, rightside_rgb, rightside_timestamp = rightside_arm_landmarks_queue.get()

            #print("oppo_timestamp: ", oppo_timestamp)
            #print("rightside_timestamp: ", rightside_timestamp)
            

            #opposite_rgb = opposite_rgb.numpy_view()[..., ::-1]
            #rightside_rgb = rightside_rgb.numpy_view()[..., ::-1]

            #opposite_arm_landmarks = get_normalized_pose_landmarks(opposite_result)
            #rightside_arm_landmarks = get_normalized_pose_landmarks(rightside_result)

            #if is_draw_landmarks:
                #opposite_rgb = draw_landmarks_on_image(opposite_rgb, opposite_arm_landmarks)
                #rightside_rgb = draw_landmarks_on_image(rightside_rgb, rightside_arm_landmarks)

        if (not opposite_streamed_frame_queue.empty() and
            not rightside_streamed_frame_queue.empty()):
        
            opposite_rgb, opposite_depth = opposite_streamed_frame_queue.get()
            rightside_rgb, rightside_depth = rightside_streamed_frame_queue.get()

            # 1. Get xyZ
            if (not opposite_arm_landmarks_queue.empty() and
                not rightside_arm_landmarks_queue.empty()):

                opposite_arm_landmark_results = opposite_arm_landmarks_queue.get()
                rightside_arm_landmark_results = rightside_arm_landmarks_queue.get()

                opposite_arm_landmarks = get_normalized_pose_landmarks(opposite_arm_landmark_results)
                rightside_arm_landmarks = get_normalized_pose_landmarks(rightside_arm_landmark_results)

                if is_draw_landmarks:
                    opposite_rgb = draw_landmarks_on_image(opposite_rgb, 
                                                           opposite_arm_landmarks)
                    rightside_rgb = draw_landmarks_on_image(rightside_rgb,
                                                            rightside_arm_landmarks)

                if num_pose == 1:
                    opposite_arm_landmarks = opposite_arm_landmarks[0]                                             
                    rightside_arm_landmarks = rightside_arm_landmarks[0] 

                landmarks_name_want_to_get = get_landmarks_name_based_on_arm(arm_to_get)
                landmarks_id_want_to_get = [pose_landmark_names.index(name) for name in landmarks_name_want_to_get]

                opposite_arm_xyZ = get_xyZ(opposite_arm_landmarks, 
                                           opposite_depth, 
                                           frame_size, 
                                           sliding_window_size,
                                           landmarks_id_want_to_get,
                                           visibility_threshold)  # shape: (N, 3)
                rightside_arm_xyZ = get_xyZ(rightside_arm_landmarks, 
                                            rightside_depth, 
                                            frame_size, 
                                            sliding_window_size,
                                            landmarks_id_want_to_get,
                                            visibility_threshold)  # shape: (N, 3)

                if (opposite_arm_xyZ is None or
                    rightside_arm_xyZ is None or
                    np.isnan(opposite_arm_xyZ).any() or 
                    np.isnan(rightside_arm_xyZ).any() or 
                    len(opposite_arm_xyZ) != len(rightside_arm_xyZ) or 
                    len(opposite_arm_xyZ) == 0 or
                    len(rightside_arm_xyZ) == 0):
                    #cv2.imshow("Frame", frame_of_two_cam)
                    continue

                # 2. Fusing landmarks of two cameras
                fused_XYZ = fuse_landmarks_from_two_cameras(opposite_arm_xyZ, rightside_arm_xyZ,  # (N, 3)
                                                            oak_ins,
                                                            rs_ins,
                                                            oak_2_rs_mat_avg) 

                if fused_XYZ.shape[0] != len(landmarks_name_want_to_get):
                    #cv2.imshow("Frame", frame_of_two_cam)
                    continue

                # 3. Convert to shoulder coord
                XYZ_wrt_shoulder = convert_to_shoulder_coord(fused_XYZ, landmarks_name_want_to_get)  # (N, 3)

                # 4. Calculate angles
                angles = get_angles_between_joints(XYZ_wrt_shoulder, landmarks_name_want_to_get)

                # 5. Plot (optional)
                if plot_3d:
                    visualization_queue.put(XYZ_wrt_shoulder)
                    if visualization_queue.qsize() > 1:
                        visualization_queue.get()

                # 6. Save landmarks (optional)
                if save_landmarks:
                    write_queue.put(XYZ_wrt_shoulder)
                    if write_queue.qsize() > 1:
                        write_queue.get()

            frame_count += 1
            elapsed_time = time.time() - start_time

            # Update FPS every second
            if elapsed_time > 1.0:
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

        time.sleep(0.5)
