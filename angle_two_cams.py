import time
import cv2
import depthai as dai
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
import threading
import queue
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import euclidean
from scipy.optimize import minimize
from functools import partial

# Initialize MediaPipe Hand solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Apply bilateral filter
d = 12            # Diameter of each pixel neighborhood
sigma_color = 25 # Filter sigma in the color space
sigma_space = 25 # Filter sigma in the coordinate space

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

# ------------------------- JOINTS' NAME ------------------------- 
finger_joints_names = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]

# -------------------- DEFINE FUNCTIONS -------------------- 
def calculate_angle_between_joints(wrist_coords, fingers_landmarks_wrt_wrist):

    def angle_between(a, b, project_to=None):
        """
        project_to = None means project to "xyz"
        """

        if a.ndim == 1:
            a = np.full_like(b, a)

        mask = [1, 1, 1]
        if project_to is not None:
            assert project_to in ["xy", "xz", "yz"]
            if project_to == "xy":
                #mask = [1, 1, 0]
                a = np.delete(a, -1, axis=1)
                b = np.delete(b, -1, axis=1)
            elif project_to == "xz":
                ##mask = [1, 0, 1]
                a = np.delete(a, 1, axis=1)
                b = np.delete(b, 1, axis=1)
            else:
                #mask = [0, 1, 1]
                a = np.delete(a, 0, axis=1)
                b = np.delete(b, 0, axis=1)
        #a = a * mask
        #b = b * mask

        dot_product = np.sum(a * b, axis=1)  # calculate dot product by element-wise style instead of using np.dot
        magnitude_a = np.linalg.norm(a, axis=1)
        magnitude_b = np.linalg.norm(b, axis=1)
        cos_theta = dot_product / (magnitude_a * magnitude_b)
        angle_radians = np.arccos(cos_theta)
        angle_degrees = np.degrees(angle_radians)

        # Get the direction
        M = np.concatenate([a[:, None, :], b[:, None, :]], axis=1)
        dets = np.linalg.det(M)
        directions = np.sign(dets)
        angle_degrees *= directions
        
        return angle_degrees

    assert np.sum(np.abs(wrist_coords)).astype(np.int8) == 0

    angles = np.zeros(shape=fingers_landmarks_wrt_wrist.shape[:-1])
    vector_y = fingers_landmarks_wrt_wrist[2, 0, :]

    """
    For now, we dont calculate the angles of thumb finger
    """

    # Angles of J11 - > J51
    # The order of params a and b is important here, because we will compute determinant to get the direction
    angles[1:, 0] = angle_between(vector_y,
                                  fingers_landmarks_wrt_wrist[1:, 1, :] - fingers_landmarks_wrt_wrist[1:, 0, :], 
                                  project_to="yz")

    # Angles of J12 - > J52
    # The order of params a and b is important here, because we will compute determinant to get the direction
    angles[1:, 1] = angle_between(fingers_landmarks_wrt_wrist[1:, 0, :],
                                  fingers_landmarks_wrt_wrist[1:, 1, :] - fingers_landmarks_wrt_wrist[1:, 0, :], 
                                  project_to="xy")

    # Angles of J13 - > J53
    # The order of params a and b is important here, because we will compute determinant to get the direction
    angles[1:, 2] = angle_between(fingers_landmarks_wrt_wrist[1:, 1, :] - fingers_landmarks_wrt_wrist[1:, 0, :], 
                                  fingers_landmarks_wrt_wrist[1:, 2, :] - fingers_landmarks_wrt_wrist[1:, 1, :], 
                                  project_to="xy")

    # Angles of J14 - > J54
    # The order of params a and b is important here, because we will compute determinant to get the direction
    angles[1:, 3] = angle_between(fingers_landmarks_wrt_wrist[1:, 2, :] - fingers_landmarks_wrt_wrist[1:, 1, :], 
                                  fingers_landmarks_wrt_wrist[1:, 3, :] - fingers_landmarks_wrt_wrist[1:, 2, :], 
                                  project_to="xy")

    return angles

def plot_3d(origin_coords, x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(origin_coords[0], origin_coords[1], origin_coords[2], c='r', marker='*')
    ax.scatter(x.flatten(), y.flatten(), z.flatten(), c='b', marker='o')
    
    # Draw a line
    def draw_line(ax, prev_p, next_p):
        ax.plot([prev_p[0], next_p[0]], [prev_p[1], next_p[1]], [prev_p[2], next_p[2]], c='b')
        return ax

    for finger_i in range(x.shape[0]):
        for joint_j in range(x.shape[1]):
            if joint_j == 0:
                prev_p = origin_coords
            else:
                prev_p = next_p
            next_p = [x[finger_i, joint_j], y[finger_i, joint_j], z[finger_i, joint_j]]
            ax = draw_line(ax, prev_p, next_p)

    # Draw Oxyz coord
    #ax.plot([-20, 200], [0, 0], [0, 0], c='r')
    #ax.plot([0, 0], [0, 200], [0, 0], c='g')
    #ax.plot([0, 0], [0, 0], [-20, 200], c='b')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the limits for each axis
    #ax.set_xlim(-20, 150)
    #ax.set_ylim(0, 200)
    #ax.set_zlim(-20, 100)  # Setting custom limits for z-axis

    fig.savefig('view_1.png')

    ax.view_init(elev=11, azim=-2)
    fig.savefig('view_2.png')
    

    ax.view_init(elev=90, azim=-90)
    fig.savefig('view_3.png')
    
    view_1 = cv2.imread('view_1.png')
    view_2 = cv2.imread('view_2.png')
    view_3 = cv2.imread('view_3.png')
    
    cv2.imshow("View 1", view_1)
    #cv2.moveWindow("View 1", window_sizes[1][0], window_sizes[1][1])
    
    cv2.imshow("View 2", view_2)
    #cv2.moveWindow("View 2", window_sizes[2][0], window_sizes[2][1])
    
    cv2.imshow("View 3", view_3)
    #cv2.moveWindow("View 3", window_sizes[3][0], window_sizes[3][1])

    plt.close()

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

def distance(Z, oak_xyZ, rs_xyZ, oak_ins, rs_ins):
    oak_Z, rs_Z = Z
    oak_XYZ = np.zeros_like(oak_xyZ)
    rs_XYZ = np.zeros_like(rs_xyZ)
    
    oak_XYZ[0] = (oak_xyZ[0] - oak_ins[0, -1]) * oak_Z / oak_ins[0, 0]
    oak_XYZ[1] = (oak_xyZ[1] - oak_ins[1, -1]) * oak_Z / oak_ins[1, 1]
    oak_XYZ[-1] = oak_Z

    rs_XYZ[0] = (rs_xyZ[0] - rs_ins[0, -1]) * rs_Z / rs_ins[0, 0]
    rs_XYZ[1] = (rs_xyZ[1] - rs_ins[1, -1]) * rs_Z / rs_ins[1, 1]
    rs_XYZ[-1] = rs_Z

    #homo = np.ones(shape=oak_XYZ.shape[0])
    oak_XYZ_homo = np.concatenate([oak_XYZ, [1]])
    oak_XYZ_in_rs = np.matmul(oak_2_rs_mat_avg, oak_XYZ_homo.T)
    oak_XYZ_in_rs = oak_XYZ_in_rs[:-1]

    return euclidean(oak_XYZ_in_rs, rs_XYZ)

# RealSense processing function
def process_realsense(rs_queue, rs_landmarks_queue):
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

                rs_landmarks_queue.put(landmarks_xyZ)

                if rs_landmarks_queue.qsize() > 1:
                    rs_landmarks_queue.get()

                #print("RealSense xyz (hand {}): ".format(hand_num), landmarks_xyZ)

        rs_combined = np.hstack((frame_rs, depth_rs_display))
        rs_queue.put(rs_combined)

        if rs_queue.qsize() > 1:
            rs_queue.get()

# OAK-D processing function
def process_oak(oak_queue, oak_landmarks_queue):
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

                oak_landmarks_queue.put(landmarks_xyZ)

                if oak_landmarks_queue.qsize() > 1:
                    oak_landmarks_queue.get()

                #print("OAK xyz (hand {}): ".format(hand_num), landmarks_xyZ)

        oak_combined = np.hstack((frame_oak, depth_oak_display))
        oak_queue.put(oak_combined)

        if oak_queue.qsize() > 1:
            oak_queue.get()

# Create queues for frame communication
rs_queue = queue.Queue()
oak_queue = queue.Queue()

# Create queues for landmarks communication
rs_landmarks_queue = queue.Queue()
oak_landmarks_queue = queue.Queue()

# Create a queue for plot data
plot_queue = queue.Queue()

# Start RealSense and OAK-D processing threads
rs_thread = threading.Thread(target=process_realsense, args=(rs_queue, rs_landmarks_queue,), daemon=True)
oak_thread = threading.Thread(target=process_oak, args=(oak_queue, oak_landmarks_queue,), daemon=True)

rs_thread.start()
oak_thread.start()

while True:
    #if not rs_queue.empty():
        #rs_combined = rs_queue.get()
        #cv2.imshow("RealSense Combined", rs_combined)
        
    #if not oak_queue.empty():
        #oak_combined = oak_queue.get()
        #cv2.imshow("OAK-D Combined", oak_combined)

    if (not rs_queue.empty()) and (not oak_queue.empty()):

        # TODO 
        # 1. Get oak_xyZ, rs_xyZ
        # 2. Fuse to fused_landmarks
        # 3. Convert to wrist coord.
        # 4. Calculate angles
        # 5. Show points if needed

        # 1. Get oak_xyZ, rs_xyZ
        rs_combined = rs_queue.get()
        oak_combined = oak_queue.get()

        rs_xyZ = rs_landmarks_queue.get()
        oak_xyZ = oak_landmarks_queue.get()
        #print("OAK xyz: ", oak_xyZ)
        #print("RS xyz: ", rs_xyZ)

        # Check NaN
        if np.isnan(rs_xyZ).any() or np.isnan(oak_xyZ).any():
            #print('NAN FOUND!!!')
            cv2.imshow("OAK-D Combined", oak_combined)
            cv2.imshow("RealSense Combined", rs_combined)
            continue

        #print("OAK xyz: ", oak_xyZ)
        #print("RS xyz: ", rs_xyZ)

        # 2. Fuse to fused_landmarks
        oak_new_Z, rs_new_Z = [], []
        for i in range(oak_xyZ.shape[0]):
            oak_i_xyZ, rs_i_xyZ = oak_xyZ[i], rs_xyZ[i]

            min_dis = partial(distance, oak_xyZ=oak_i_xyZ, rs_xyZ=rs_i_xyZ, oak_ins=oak_ins, rs_ins=rs_ins)
            result = minimize(min_dis, x0=[oak_i_xyZ[-1], rs_i_xyZ[-1]])
            oak_i_new_Z, rs_i_new_Z = result.x
            oak_new_Z.append(oak_i_new_Z)
            rs_new_Z.append(rs_i_new_Z)

        oak_new_xyZ = oak_xyZ.copy()
        rs_new_xyZ = rs_xyZ.copy()

        oak_new_xyZ[:, -1] = oak_new_Z
        rs_new_xyZ[:, -1] = rs_new_Z
        oak_new_XYZ = np.zeros_like(oak_new_xyZ)
        rs_new_XYZ = np.zeros_like(rs_new_xyZ)
        oak_new_XYZ[:, 0] = (oak_new_xyZ[:, 0] - oak_ins[0, -1]) * oak_new_xyZ[:, -1] / oak_ins[0, 0]
        oak_new_XYZ[:, 1] = (oak_new_xyZ[:, 1] - oak_ins[1, -1]) * oak_new_xyZ[:, -1] / oak_ins[1, 1]
        oak_new_XYZ[:, -1] = oak_new_xyZ[:, -1]
        rs_new_XYZ[:, 0] = (rs_new_xyZ[:, 0] - rs_ins[0, -1]) * rs_new_xyZ[:, -1] / rs_ins[0, 0]
        rs_new_XYZ[:, 1] = (rs_new_xyZ[:, 1] - rs_ins[1, -1]) * rs_new_xyZ[:, -1] / rs_ins[1, 1]
        rs_new_XYZ[:, -1] = rs_new_xyZ[:, -1]
        fused_landmarks = (oak_new_XYZ + rs_new_XYZ) / 2

        # 3. Convert to wrist coord.
        u = fused_landmarks[finger_joints_names.index("INDEX_FINGER_MCP"), :] - fused_landmarks[finger_joints_names.index("WRIST"), :]
        y = fused_landmarks[finger_joints_names.index("MIDDLE_FINGER_MCP"), :] - fused_landmarks[finger_joints_names.index("WRIST"), :]

        x = np.cross(y, u)
        z = np.cross(x, y)

        x = x / np.linalg.norm(x)
        y = y / np.linalg.norm(y)
        z = z / np.linalg.norm(z)

        w_c = fused_landmarks[finger_joints_names.index("WRIST"), :]

        R = np.array([x, y, z, w_c])
        R = np.concatenate([R, np.expand_dims([0, 0, 0, 1], 1)], axis=1)
        R = np.transpose(R)
        R_inv = np.linalg.inv(R)
        homo = np.ones(shape=fused_landmarks.shape[0])
        fused_landmarks = np.concatenate([fused_landmarks, np.expand_dims(homo, 1)], axis=1)
        landmarks_wrt_wrist = np.matmul(R_inv, fused_landmarks.T)
        landmarks_wrt_wrist = landmarks_wrt_wrist.T
        wrist_coords, fingers_landmarks_wrt_wrist = landmarks_wrt_wrist[0, :-1], landmarks_wrt_wrist[1:, :-1]
        fingers_landmarks_wrt_wrist = fingers_landmarks_wrt_wrist.reshape(5, 4, 3)

        # 4. Calculate angles
        angles = calculate_angle_between_joints(wrist_coords, fingers_landmarks_wrt_wrist)
        print(angles)
        
        # 5. Plot hand
        #plot_3d(wrist_coords,
                #fingers_landmarks_wrt_wrist[:, :, 0], 
                #fingers_landmarks_wrt_wrist[:, :, 1], 
                #fingers_landmarks_wrt_wrist[:, :, 2])

        # 6. Show image
        cv2.imshow("OAK-D Combined", oak_combined)
        cv2.imshow("RealSense Combined", rs_combined)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()