import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import threading
import math
import matplotlib.pyplot as plt
import pyrealsense2 as rs

window_sizes = [(0, 0), (650, 0), (1300, 0), (650, 550)]
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def hand_tracking(color_image,
                  depth_image,
                  fx,
                  fy,
                  ox,
                  oy,
                  window_size=(640, 480)):
    """
    Process:
        1. Detect landmarks (x, y).
        2. Unnormalize (x_unnorm, y_unnorm) by frame_width, frame_height.
        3. Get z from depth_image.
        4. Currently, (x_unnorm, y_unnorm) is in (px) unit. So convert it into (mm) unit.
        5. Transform all landmarks into wrist coordinate.
        6. Calculate angles between two adjacent joints and get the direction of the angle.
    """

    processed_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    wrist_landmarks = None
        
    results = hands.process(processed_image)
    if results.multi_hand_landmarks:
        for num, hand in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(color_image, hand, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))
            
    return color_image, None

if __name__ == "__main__":
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    window_size = (640, 480)

    config.enable_stream(rs.stream.depth, window_size[0], window_size[1], rs.format.z16, 30)
    config.enable_stream(rs.stream.color, window_size[0], window_size[1], rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    _profile = pipeline.get_active_profile()
    profile = _profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = profile.get_intrinsics()
    intrinsics = np.array(
        [[intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1]]
    )

    rsalign = rs.align(rs.stream.color)

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        try:
            while True:

                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                frames = rsalign.process(frames) if rsalign else frames

                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not color_frame or not depth_frame:
                    continue

                ## Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                #depth_image = cv2.flip(depth_image, 1)
                #color_image = cv2.flip(color_image, 1)

                ## Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                color_image, wrist_landmarks = hand_tracking(color_image=color_image.copy(), 
                                                             depth_image=depth_image.copy(),
                                                             fx=intrinsics[0, 0],
                                                             fy=intrinsics[1, 1],
                                                             ox=intrinsics[0, -1],
                                                             oy=intrinsics[1, -1],
                                                             window_size=window_size)

                depth_colormap_dim = depth_colormap.shape
                color_colormap_dim = color_image.shape

                # If depth and color resolutions are different, resize color image to match depth image for display
                if depth_colormap_dim != color_colormap_dim:
                    resized_color_image = cv2.resize(color_image, 
                                                    dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), 
                                                    interpolation=cv2.INTER_AREA)
                    images = np.hstack((resized_color_image, depth_colormap))
                else:
                    images = np.hstack((color_image, depth_colormap))

                # Show images
                cv2.namedWindow('Hand Tracking', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Hand Tracking', images)

                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break

        finally:
            # Stop streaming
            pipeline.stop()
            cv2.destroyAllWindows()