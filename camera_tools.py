import depthai as dai
import pyrealsense2 as rs
import numpy as np
import cv2
import time

def initialize_realsense_cam(rgb_size, depth_size):
    pipeline_rs = rs.pipeline()
    config_rs = rs.config()
    config_rs.enable_stream(rs.stream.color, rgb_size[0], rgb_size[1], rs.format.bgr8, 30)
    config_rs.enable_stream(rs.stream.depth, depth_size[0], depth_size[1], rs.format.z16, 30)

    # Start streaming
    pipeline_rs.start(config_rs)
    rsalign = rs.align(rs.stream.color)

    return pipeline_rs, rsalign

def initialize_oak_cam(stereo_size):
    pipeline_oak = dai.Pipeline()

    cam_rgb = pipeline_oak.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setFps(30)

    mono_left = pipeline_oak.create(dai.node.MonoCamera)
    mono_right = pipeline_oak.create(dai.node.MonoCamera)
    stereo = pipeline_oak.create(dai.node.StereoDepth)

    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    stereo.setOutputSize(stereo_size[0], stereo_size[1])

    mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    xout_rgb = pipeline_oak.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.video.link(xout_rgb.input)

    xout_depth = pipeline_oak.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    return pipeline_oak

def stream_rs(pipeline_rs, rsalign, rs_frame_queue):
    while True:
        frames = pipeline_rs.wait_for_frames()
        frames = rsalign.process(frames)
        color_frame_rs = frames.get_color_frame()
        depth_frame_rs = frames.get_depth_frame()

        frame_rs = np.asanyarray(color_frame_rs.get_data())
        depth_rs = np.asanyarray(depth_frame_rs.get_data(), dtype=np.float32)

        rs_frame_queue.put((frame_rs, depth_rs))

        if rs_frame_queue.qsize() > 1:
            rs_frame_queue.get()

        time.sleep(0.001)

def stream_oak(pipeline_oak, oak_frame_queue, mxid=None):
    if mxid is not None:
        device_info = dai.DeviceInfo(mxid) 
        device_oak = dai.Device(pipeline_oak, device_info)
    else:
        device_oak = dai.Device(pipeline_oak)
    rgb_queue_oak = device_oak.getOutputQueue(name="rgb", maxSize=2, blocking=False)
    depth_queue_oak = device_oak.getOutputQueue(name="depth", maxSize=2, blocking=False)

    while True:
        rgb_frame_oak = rgb_queue_oak.get()
        depth_frame_oak = depth_queue_oak.get()

        frame_oak = rgb_frame_oak.getCvFrame()
        depth_oak = depth_frame_oak.getFrame()

        oak_frame_queue.put((frame_oak, depth_oak))

        if oak_frame_queue.qsize() > 1:
            oak_frame_queue.get()

        time.sleep(0.001)