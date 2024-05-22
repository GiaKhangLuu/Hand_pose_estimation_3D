import cv2
import depthai as dai
import mediapipe as mp
import numpy as np

# Create pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setFps(30)

# Define depth camera
mono_left = pipeline.create(dai.node.MonoCamera)
mono_right = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

# Link mono cameras to stereo depth
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

# XLinkOut for RGB and depth
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.video.link(xout_rgb.input)

xout_depth = pipeline.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

# Initialize MediaPipe Hand solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Connect to the device and start the pipeline
with dai.Device(pipeline) as device:
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    depth_queue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    while True:
        rgb_frame = rgb_queue.get()
        depth_frame = depth_queue.get()

        frame = rgb_frame.getCvFrame()
        depth = depth_frame.getFrame()
        depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth = cv2.applyColorMap(depth.astype(np.uint8), cv2.COLORMAP_JET)

        # Resize the frame for a smaller window
        frame = cv2.resize(frame, (640, 480))
        depth = cv2.resize(depth, (640, 480))

        # Convert the frame to RGB (MediaPipe expects RGB input)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
        results = hands.process(frame_rgb)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get hand landmark points and depth
                #for landmark in hand_landmarks.landmark:
                    #x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    #if 0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]:
                        #depth_value = depth_frame.getFrame()[y, x]
                        #cv2.putText(frame, f'{depth_value:.2f}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        # Horizontally stack the RGB frame and depth map
        combined_frame = cv2.hconcat([frame, depth])

        # Display the combined frame
        cv2.imshow("Hand Tracking and Depth Map", combined_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
