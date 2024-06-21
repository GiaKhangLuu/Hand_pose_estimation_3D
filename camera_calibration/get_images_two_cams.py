import cv2
import depthai as dai
import numpy as np
import pyrealsense2 as rs
import threading
import queue
import os
import shutil

# RealSense processing function
def process_realsense(rs_queue):
    pipeline_rs = rs.pipeline()
    config_rs = rs.config()
    config_rs.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config_rs.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    pipeline_rs.start(config_rs)

    while True:
        frames = pipeline_rs.wait_for_frames()
        color_frame_rs = frames.get_color_frame()
        #depth_frame_rs = frames.get_depth_frame()
        if not color_frame_rs:
            continue

        frame_rs = np.asanyarray(color_frame_rs.get_data())

        rs_queue.put(frame_rs)

        if rs_queue.qsize() > 1:
            rs_queue.get()

# OAK-D processing function
def process_oak(oak_queue):
    pipeline_oak = dai.Pipeline()

    cam_rgb = pipeline_oak.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setFps(30)

    xout_rgb = pipeline_oak.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.video.link(xout_rgb.input)

    device_oak = dai.Device(pipeline_oak)
    rgb_queue_oak = device_oak.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        rgb_frame_oak = rgb_queue_oak.get()

        frame_oak = rgb_frame_oak.getCvFrame()

        frame_oak = cv2.resize(frame_oak, (1280, 720))
        #depth_oak_display = cv2.resize(depth_oak, (640, 480))

        oak_queue.put(frame_oak)

        if oak_queue.qsize() > 1:
            oak_queue.get()

# Create queues for frame communication
rs_queue = queue.Queue(maxsize=1)
oak_queue = queue.Queue(maxsize=1)

# Start RealSense and OAK-D processing threads
rs_thread = threading.Thread(target=process_realsense, args=(rs_queue,))
oak_thread = threading.Thread(target=process_oak, args=(oak_queue,))

rs_thread.start()
oak_thread.start()

folder_path = './images'
rs_path = './images/rs'
oak_path = './images/oak'


if os.path.exists(folder_path):
    shutil.rmtree(folder_path)
os.makedirs(folder_path)

if not os.path.exists(rs_path):
    os.makedirs(rs_path)

if not os.path.exists(oak_path):
    os.makedirs(oak_path)

num = 0

while True:

    if (not rs_queue.empty()) and (not oak_queue.empty()):
        rs_frame = rs_queue.get()
        cv2.imshow("RealSense Combined", rs_frame)
        
        oak_frame = oak_queue.get()
        cv2.imshow("OAK-D Combined", oak_frame)

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('{}/rs_img_'.format(rs_path) + str(num) + '.png', rs_frame)
        cv2.imwrite('{}/oak_img_'.format(oak_path) + str(num) + '.png', oak_frame)
        print("image saved!")
        num += 1

cv2.destroyAllWindows()
