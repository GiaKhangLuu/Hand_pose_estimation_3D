# for creating a responsive plot 
#%matplotlib inline
#%matplotlib widget 

import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import threading
import math
import matplotlib.pyplot as plt
import pyrealsense2 as rs
from mpl_toolkits.mplot3d import Axes3D

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

finger_joints_names = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]

window_sizes = [(0, 0), (650, 0), (1300, 0), (650, 550)]

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
    ax.plot([-20, 200], [0, 0], [0, 0], c='r')
    ax.plot([0, 0], [0, 200], [0, 0], c='g')
    ax.plot([0, 0], [0, 0], [-20, 200], c='b')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the limits for each axis
    ax.set_xlim(-20, 150)
    ax.set_ylim(0, 200)
    ax.set_zlim(-20, 100)  # Setting custom limits for z-axis

    fig.savefig('view_1.png')

    ax.view_init(elev=11, azim=-2)
    fig.savefig('view_2.png')
    

    ax.view_init(elev=90, azim=-90)
    fig.savefig('view_3.png')
    
    view_1 = cv2.imread('view_1.png')
    view_2 = cv2.imread('view_2.png')
    view_3 = cv2.imread('view_3.png')
    
    cv2.imshow("View 1", view_1)
    cv2.moveWindow("View 1", window_sizes[1][0], window_sizes[1][1])
    
    cv2.imshow("View 2", view_2)
    cv2.moveWindow("View 2", window_sizes[2][0], window_sizes[2][1])
    
    cv2.imshow("View 3", view_3)
    cv2.moveWindow("View 3", window_sizes[3][0], window_sizes[3][1])

    plt.close()
    #return ax

def convert_to_wrist_coordinate(landmarks, R_inv):
    homo = np.ones(shape=landmarks.shape[0])
    landmarks = np.concatenate([landmarks, np.expand_dims(homo, 1)], axis=1)
    landmarks_wrt_wrist = np.matmul(R_inv, np.transpose(landmarks))
    landmarks_wrt_wrist = np.transpose(landmarks_wrt_wrist)

    return landmarks_wrt_wrist

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

def get_depth(xy_landmarks, depth_image):
    xy_landmarks = xy_landmarks.astype(np.int32)
    z_landmarks = depth_image[xy_landmarks[:, 1], xy_landmarks[:, 0]]
    
    return z_landmarks

def convert_xy_from_px_to_mm(landmarks_xy_unnorm, landmarks_z, fx, fy, ox, oy):
    xy_mm = landmarks_xy_unnorm
    xy_mm[:, 0] = (xy_mm[:, 0] - ox) * landmarks_z / fx
    xy_mm[:, 1] = (xy_mm[:, 1] - oy) * landmarks_z / fy

    return xy_mm

def get_transformation_matrix_to_wrist_coordinate(landmarks_xyz):
    u = landmarks_xyz[finger_joints_names.index("INDEX_FINGER_MCP"), :] - landmarks_xyz[finger_joints_names.index("WRIST"), :]
    y = landmarks_xyz[finger_joints_names.index("MIDDLE_FINGER_MCP"), :] - landmarks_xyz[finger_joints_names.index("WRIST"), :]

    x = np.cross(y, u)
    z = np.cross(x, y)

    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    z = z / np.linalg.norm(z)

    w_c = landmarks_xyz[finger_joints_names.index("WRIST"), :]

    R = np.array([x, y, z, w_c])
    R = np.concatenate([R, np.expand_dims([0, 0, 0, 1], 1)], axis=1)
    R = np.transpose(R)
    R_inv = np.linalg.inv(R)

    return R_inv

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

            """             PRINT VALUES
            #for i, landmarks in enumerate(hand.landmark):
                ##if landmarks.z < 0:
                    ##print(finger_joints_names[i])
                    ##print(landmarks)
                ##if i not in [8, 12]:
                    ##continue
                #coords = tuple(np.multiply(np.array((landmarks.x, 
                                                     #landmarks.y)), 
                                        #[640, 480]).astype(int))  
                #print((landmarks.x, landmarks.y, landmarks.z))
                #image = cv2.putText(image, str(round(landmarks.z, 2)), coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
            """

            raw_landmarks = []
            for landmark in hand.landmark:
                x = landmark.x 
                y = landmark.y
                z = landmark.z
                raw_landmarks.append([x, y, z]) 
            raw_landmarks = np.array(raw_landmarks)

            print("index finger raw: ", np.around(raw_landmarks[5:9, :], decimals=2))

            landmarks_xy_unnorm = unnormalize_landmarks(raw_landmarks, window_size)

            landmarks_z = get_depth(landmarks_xy_unnorm, depth_image)

            print("depth of finger raw: ", landmarks_z[5:9])

            landmarks_xy_mm = convert_xy_from_px_to_mm(landmarks_xy_unnorm, landmarks_z, fx, fy, ox, oy)

            landmarks_xyz = np.concatenate([landmarks_xy_mm, np.expand_dims(landmarks_z, -1)], axis=-1)

            R_inv = get_transformation_matrix_to_wrist_coordinate(landmarks_xyz) 
            
            landmarks_wrt_wrist = convert_to_wrist_coordinate(landmarks_xyz, R_inv)  # landmarks_wrt_wrist.shape = (21, 3)

            wrist_coords, fingers_landmarks_wrt_wrist = landmarks_wrt_wrist[0, :-1], landmarks_wrt_wrist[1:, :-1]
            fingers_landmarks_wrt_wrist = fingers_landmarks_wrt_wrist.reshape(5, 4, 3)  # new shape = (5, 4, 3); 5 fingers, 4 joints, 3 values

            print("index finger wrt wrist: ", np.around(fingers_landmarks_wrt_wrist[1, :, :], decimals=2))
            #plot_3d(wrist_coords,
                    #fingers_landmarks_wrt_wrist[1:, :, 0], 
                    #fingers_landmarks_wrt_wrist[1:, :, 1], 
                    #fingers_landmarks_wrt_wrist[1:, :, 2])

            angles = calculate_angle_between_joints(wrist_coords, fingers_landmarks_wrt_wrist)
            #print(angles)
            #print(fingers_landmarks_wrt_wrist)

            #image = cv2.putText(color_image, "J24: {}".format(str(round(angles[1, 3], 2))), (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            #image = cv2.putText(color_image, "J34: {}".format(str(round(angles[2, 3], 2))), (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            #image = cv2.putText(color_image, "J44: {}".format(str(round(angles[3, 3], 2))), (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            #image = cv2.putText(color_image, "J54: {}".format(str(round(angles[4, 3], 2))), (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
    return color_image, None

def thread_show_image(angles=None):
    #global angles
    while True:
        if angles is not None:
            plot_3d(wrist_coords,
                    fingers_landmarks_wrt_wrist[:, :, 0], 
                    fingers_landmarks_wrt_wrist[:, :, 1], 
                    fingers_landmarks_wrt_wrist[:, :, 2])

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

                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                #depth_image = cv2.flip(depth_image, 1)
                #color_image = cv2.flip(color_image, 1)

                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
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
                cv2.moveWindow("Hand Tracking", window_sizes[0][0], window_sizes[0][1])

                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break

        finally:
            # Stop streaming
            pipeline.stop()
            cv2.destroyAllWindows()