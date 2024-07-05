import yaml
import cv2
import numpy as np
import mediapipe as mp
import time
from functools import partial
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import euclidean

from numpy.typing import NDArray
from typing import Tuple, List

from mediapipe.framework.formats import landmark_pb2
import numpy as np

def filter_depth(depth_array: NDArray, sliding_window_size, sigma_color, sigma_space) -> NDArray:
    depth_array = depth_array.astype(np.float32)
    depth_array = cv2.bilateralFilter(depth_array, sliding_window_size, sigma_color, sigma_space)
    return depth_array

def detect_arm_landmarks(rs_detector, oak_detector, input_queue, result_queue, image_format="bgr"):
    while True:
        if not input_queue.empty():
            rs_color_img, oak_color_img, timestamp = input_queue.get()

            processed_rs_img = np.copy(rs_color_img) 
            processed_oak_img = np.copy(oak_color_img)
            if image_format == "bgr":
                processed_rs_img = cv2.cvtColor(processed_rs_img, cv2.COLOR_BGR2RGB)
                processed_oak_img = cv2.cvtColor(processed_oak_img, cv2.COLOR_BGR2RGB)

            mp_rs_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=processed_rs_img)
            mp_oak_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=processed_oak_img)

            rs_result = rs_detector.detect(mp_rs_image)
            oak_result = oak_detector.detect(mp_oak_image)

            if rs_result.pose_landmarks and oak_result.pose_landmarks:
                result_queue.put((rs_result, oak_result))
                if result_queue.qsize() > 1:
                    result_queue.get()

        time.sleep(0.0001)

def detect_hand_landmarks(rs_detector, oak_detector, input_queue, result_queue, image_format="bgr"):
    while True:
        if not input_queue.empty():
            rs_color_img, oak_color_img, timestamp = input_queue.get()

            processed_rs_img = np.copy(rs_color_img) 
            processed_oak_img = np.copy(oak_color_img)
            if image_format == "bgr":
                processed_rs_img = cv2.cvtColor(processed_rs_img, cv2.COLOR_BGR2RGB)
                processed_oak_img = cv2.cvtColor(processed_oak_img, cv2.COLOR_BGR2RGB)

            mp_rs_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=processed_rs_img)
            mp_oak_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=processed_oak_img)

            rs_result = rs_detector.detect_for_video(mp_rs_image, timestamp)
            oak_result = oak_detector.detect_for_video(mp_oak_image, timestamp)

            if rs_result.hand_landmarks and oak_result.hand_landmarks:
                result_queue.put((rs_result, oak_result))
                if result_queue.qsize() > 1:
                    result_queue.get()

        time.sleep(0.0001)