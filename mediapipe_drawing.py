from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
from typing import Tuple, List

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_arm_landmarks_on_image(rgb_image, 
	pose_landmarks_proto_list: List[landmark_pb2.NormalizedLandmarkList]):
	annotated_image = np.copy(rgb_image)

	# Loop through the detected poses to visualize.
	for pose_landmarks_proto in pose_landmarks_proto_list:
		solutions.drawing_utils.draw_landmarks(
    		annotated_image,
    		pose_landmarks_proto,
    		solutions.pose.POSE_CONNECTIONS,
    		solutions.drawing_styles.get_default_pose_landmarks_style())
	return annotated_image

def draw_hand_landmarks_on_image(rgb_image, 
	hand_landmarks_proto_list: List[landmark_pb2.NormalizedLandmark],
	handedness: List):
	annotated_image = np.copy(rgb_image)

	# Loop through the detected hands to visualize.
	for hand, hand_landmarks_proto in zip(handedness, hand_landmarks_proto_list):
		solutions.drawing_utils.draw_landmarks(
    		annotated_image,
    		hand_landmarks_proto,
    		solutions.hands.HAND_CONNECTIONS,
    		solutions.drawing_styles.get_default_hand_landmarks_style(),
    		solutions.drawing_styles.get_default_hand_connections_style())

		# Get the top left corner of the detected hand's bounding box.
		height, width, _ = annotated_image.shape

		x_coordinates = [landmark.x for landmark in hand_landmarks_proto.landmark]
		y_coordinates = [landmark.y for landmark in hand_landmarks_proto.landmark]
		text_x = int(min(x_coordinates) * width)
		text_y = int(min(y_coordinates) * height) - MARGIN

		# Draw handedness (left or right hand) on the image.
		cv2.putText(annotated_image, f"{hand}",
            		(text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
            		FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

	return annotated_image