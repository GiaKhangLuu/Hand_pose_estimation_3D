import cv2
import depthai as dai
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
import threading
import queue

# Initialize MediaPipe Hand solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Apply bilateral filter
d = 12            # Diameter of each pixel neighborhood
sigma_color = 25 # Filter sigma in the color space
sigma_space = 25 # Filter sigma in the coordinate space

def unnormalize_landmarks(landmarks, window_size):
    unnorm_landmarks = landmarks[:, :-1]
    unnorm_landmarks[:, 0] = unnorm_landmarks[:, 0] * window_size[0]
    unnorm_landmarks[:, 1] = unnorm_landmarks[:, 1] * window_size[1]

    """
    Note: some landmarks have value > or < window_height and window_width,
        so this will cause the depth_image out of bound. For now, we just 
        clip in in the range of window's dimension. But those values 
        properly be removed from the list.
    """

    unnorm_landmarks[:, 0] = np.clip(unnorm_landmarks[:, 0], 0, window_size[0] - 1)
    unnorm_landmarks[:, 1] = np.clip(unnorm_landmarks[:, 1], 0, window_size[1] - 1)

    return unnorm_landmarks

def get_depth(xy_landmarks, depth_image, window_size):
    half_size = window_size // 2
    xy_landmarks = xy_landmarks.astype(np.int32)

    x_min = np.maximum(0, xy_landmarks[:, 0] - half_size)
    x_max = np.minimum(depth_image.shape[1] - 1, xy_landmarks[:, 0] + half_size)
    y_min = np.maximum(0, xy_landmarks[:, 1] - half_size)
    y_max = np.minimum(depth_image.shape[0] - 1, xy_landmarks[:, 1] + half_size)

    xy_windows = np.concatenate([x_min[:, None], x_max[:, None], y_min[:, None], y_max[:, None]], axis=-1)

    z_landmarks = []
    for i in range(xy_windows.shape[0]):
        z_values = depth_image[xy_windows[i, 2]:xy_windows[i, 3] + 1, xy_windows[i, 0]:xy_windows[i, 1] + 1]
        mask = z_values > 0
        z_values = z_values[mask]
        z_median = np.median(z_values)
        z_landmarks.append(z_median)

    return np.array(z_landmarks)

# RealSense processing function
def process_realsense(rs_queue):
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
    pipeline_rs = rs.pipeline()
    config_rs = rs.config()

    config_rs.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 30)
    config_rs.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
    pipeline_rs.start(config_rs)

    rsalign = rs.align(rs.stream.color)

    window_size = (640, 360)

    while True:
        frames = pipeline_rs.wait_for_frames()
        frames = rsalign.process(frames) 
        color_frame_rs = frames.get_color_frame()
        depth_frame_rs = frames.get_depth_frame()
        if not color_frame_rs or not depth_frame_rs:
            continue

        frame_rs = np.asanyarray(color_frame_rs.get_data())
        depth_rs = np.asanyarray(depth_frame_rs.get_data(), dtype=np.float32)

        depth_rs = cv2.bilateralFilter(depth_rs, d, sigma_color, sigma_space)

        depth_rs_display = cv2.normalize(depth_rs, None, 0, 255, cv2.NORM_MINMAX)
        depth_rs_display = cv2.applyColorMap(depth_rs_display.astype(np.uint8), cv2.COLORMAP_JET)

        frame_rs_rgb = cv2.cvtColor(frame_rs, cv2.COLOR_BGR2RGB)
        results_rs = hands.process(frame_rs_rgb)
        if results_rs.multi_hand_landmarks:
            for hand_num, hand_landmarks in enumerate(results_rs.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame_rs, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                raw_landmarks = []
                for landmark in hand_landmarks.landmark:
                    x = landmark.x 
                    y = landmark.y
                    z = landmark.z
                    raw_landmarks.append([x, y, z]) 
                raw_landmarks = np.array(raw_landmarks)

                #print("index finger raw: ", np.around(raw_landmarks[5:9, :], decimals=2))

                landmarks_xy_unnorm = unnormalize_landmarks(raw_landmarks, window_size)

                landmarks_Z = get_depth(landmarks_xy_unnorm, depth_rs, window_size=d)

                landmarks_xyZ = np.concatenate([landmarks_xy_unnorm, landmarks_Z[:, None]], axis=-1)

                print("RealSense xyz (hand {}): ".format(hand_num), landmarks_xyZ)

        rs_combined = np.hstack((frame_rs, depth_rs_display))
        rs_queue.put(rs_combined)

        if rs_queue.qsize() > 1:
            rs_queue.get()

# OAK-D processing function
def process_oak(oak_queue):
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
    pipeline_oak = dai.Pipeline()

    cam_rgb = pipeline_oak.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setFps(30)

    mono_left = pipeline_oak.create(dai.node.MonoCamera)
    mono_right = pipeline_oak.create(dai.node.MonoCamera)
    stereo = pipeline_oak.create(dai.node.StereoDepth)

    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    stereo.setOutputSize(640, 360)

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

    device_oak = dai.Device(pipeline_oak)
    rgb_queue_oak = device_oak.getOutputQueue(name="rgb", maxSize=2, blocking=False)
    depth_queue_oak = device_oak.getOutputQueue(name="depth", maxSize=2, blocking=False)

    window_size = (640, 360)

    while True:
        rgb_frame_oak = rgb_queue_oak.get()
        depth_frame_oak = depth_queue_oak.get()

        frame_oak = rgb_frame_oak.getCvFrame()
        depth_oak = depth_frame_oak.getFrame()
        depth_oak = depth_oak.astype(np.float32)

        frame_oak = cv2.resize(frame_oak, window_size)
        #depth_oak = cv2.resize(depth_oak, window_size)
        depth_oak = cv2.bilateralFilter(depth_oak, d, sigma_color, sigma_space)

        depth_oak_display = cv2.normalize(depth_oak, None, 0, 255, cv2.NORM_MINMAX)
        depth_oak_display = cv2.applyColorMap(depth_oak_display.astype(np.uint8), cv2.COLORMAP_JET)

        frame_oak_rgb = cv2.cvtColor(frame_oak, cv2.COLOR_BGR2RGB)
        results_oak = hands.process(frame_oak_rgb)
        if results_oak.multi_hand_landmarks:
            for hand_num, hand_landmarks in enumerate(results_oak.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame_oak, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                raw_landmarks = []
                for landmark in hand_landmarks.landmark:
                    x = landmark.x 
                    y = landmark.y
                    z = landmark.z
                    raw_landmarks.append([x, y, z]) 
                raw_landmarks = np.array(raw_landmarks)

                #print("index finger raw: ", np.around(raw_landmarks[5:9, :], decimals=2))

                landmarks_xy_unnorm = unnormalize_landmarks(raw_landmarks, window_size)

                landmarks_Z = get_depth(landmarks_xy_unnorm, depth_oak, window_size=d)

                landmarks_xyZ = np.concatenate([landmarks_xy_unnorm, landmarks_Z[:, None]], axis=-1)

                print("OAK xyz (hand {}): ".format(hand_num), landmarks_xyZ)

        oak_combined = np.hstack((frame_oak, depth_oak_display))
        oak_queue.put(oak_combined)

        if oak_queue.qsize() > 1:
            oak_queue.get()

# Create queues for frame communication
rs_queue = queue.Queue()
oak_queue = queue.Queue()

# Start RealSense and OAK-D processing threads
rs_thread = threading.Thread(target=process_realsense, args=(rs_queue,))
oak_thread = threading.Thread(target=process_oak, args=(oak_queue,))

rs_thread.start()
oak_thread.start()

while True:
    if not rs_queue.empty():
        rs_combined = rs_queue.get()
        cv2.imshow("RealSense Combined", rs_combined)
        
    if not oak_queue.empty():
        oak_combined = oak_queue.get()
        cv2.imshow("OAK-D Combined", oak_combined)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
