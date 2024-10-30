import os
import numpy as np
import cv2

from mmcv.image import imread
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import init_model as init_pose_estimator
from mmpose.apis import inference_topdown
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from mmpose.evaluation.functional import nms
from mmpose.utils import adapt_mmdet_pipeline

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

from utilities import (get_normalized_hand_landmarks,
    get_normalized_pose_landmarks,
    get_landmarks_name_based_on_arm)
from mediapipe_drawing import draw_hand_landmarks_on_image, draw_arm_landmarks_on_image
from common import (get_xyZ,
    get_depth)

from landmark_detector import RTMPoseDetector

class LandmarksDetectors:
    def __init__(
        self, 
        model_selected_id, 
        model_list, 
        model_config, 
        draw_landmarks
    ):
        """
        Desc.
        Parameters:
            attribute1 (type): Description of attribute1.
            attribute2 (type): Description of attribute2.
        """

        self._model_name = model_list[model_selected_id]
        config_by_model = model_config[self._model_name]
        hand_to_fuse = model_config["hand_to_fuse"]
        arm_to_fuse = model_config["arm_to_fuse"]
        self._num_person_to_detect = model_config["num_person_to_detect"]
        self._draw_landmarks = draw_landmarks

        assert hand_to_fuse in ["left", "right", "both"]
        assert arm_to_fuse == hand_to_fuse

        if self._model_name == "mediapipe":
            # -------------- INIT MEDIAPIPE MODELS -------------- 
            self._mediapipe_cvt_color = config_by_model["convert_color_channel"]
            self._hand_detection_activation = config_by_model["hand_detection"]["is_enable"]
            hand_min_det_conf = config_by_model["hand_detection"]["min_detection_confidence"]
            hand_min_tracking_conf = config_by_model["hand_detection"]["min_tracking_confidence"]
            hand_min_presence_conf = config_by_model["hand_detection"]["min_presence_confidence"]
            hand_num_hand_detected = config_by_model["hand_detection"]["num_hand"]
            hand_model_path = config_by_model["hand_detection"]["model_asset_path"]
            hand_landmarks_name = tuple(config_by_model["hand_detection"]["hand_landmarks"])

            self._body_detection_activation = config_by_model["body_detection"]["is_enable"]
            body_min_det_conf = config_by_model["body_detection"]["min_detection_confidence"]
            body_min_tracking_conf = config_by_model["body_detection"]["min_tracking_confidence"]
            self._body_visibility_threshold = config_by_model["body_detection"]["visibility_threshold"]
            body_model_path = config_by_model["body_detection"]["model_asset_path"]
            body_landmarks_name = tuple(config_by_model["body_detection"]["body_landmarks"])

            self._hand_to_fuse = "Left" if hand_to_fuse else "Right"  # Currently select left or right (fix this if get both left and right)
            body_landmarks_names_want_to_get = get_landmarks_name_based_on_arm(arm_to_fuse)
            self._body_landmarks_id_want_to_get = [body_landmarks_name.index(name) for name in body_landmarks_names_want_to_get]
            self._body_hand_fused_names = body_landmarks_names_want_to_get[:-1].copy()  # remove right_elbow
            self._body_hand_fused_names.extend(hand_landmarks_name)
            self._body_hand_fused_names.append(body_landmarks_names_want_to_get[-1])  # append right_elbow back

            if self._hand_detection_activation:
                hand_base_options = python.BaseOptions(model_asset_path=hand_model_path,
                    delegate=mp.tasks.BaseOptions.Delegate.GPU)
                hand_options = vision.HandLandmarkerOptions(
                    base_options=hand_base_options,
                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                    num_hands=hand_num_hand_detected,
                    min_hand_detection_confidence=hand_min_det_conf,
                    min_hand_presence_confidence=hand_min_presence_conf,
                    min_tracking_confidence=hand_min_tracking_conf)
                self._left_camera_hand_detector = vision.HandLandmarker.create_from_options(hand_options)
                self._right_camera_hand_detector = vision.HandLandmarker.create_from_options(hand_options)
            else:
                self._left_camera_hand_detector = None
                self._right_camera_hand_detector = None

            if self._body_detection_activation:
                body_base_options = python.BaseOptions(model_asset_path=body_model_path,
                    delegate=mp.tasks.BaseOptions.Delegate.GPU)
                body_options = vision.PoseLandmarkerOptions(
                    base_options=body_base_options,
                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                    num_poses=self._num_person_to_detect,  
                    min_pose_detection_confidence=body_min_det_conf,
                    min_tracking_confidence=body_min_tracking_conf,
                    output_segmentation_masks=False)
                self._left_camera_body_detector = vision.PoseLandmarker.create_from_options(body_options)  # Each camera is responsible for one detector because the detector stores the pose in prev. frame
                self._right_camera_body_detector = vision.PoseLandmarker.create_from_options(body_options)  # Each camera is responsible for one detector because the detector stores the pose in prev. frame
            else:
                self._left_camera_body_detector = None
                self._right_camera_body_detector = None
        else:
            self._rtmpose_models = RTMPoseDetector(config_by_model, self._draw_landmarks)
    
    def detect(self, img, depth_map, timestamp, side):
        assert side in ["left", "right"]
        img_size = img.shape[:-1]
        body_hand_selected_xyZ = None
        drawed_img = np.copy(img)

        if self._model_name == "mediapipe":
            # --------------  MEDIAPIPE DETECTION --------------  
            hand_detector, body_detector = None, None
            hand_landmarks, handedness = None, None
            body_landmarks = None
            body_landmarks_xyZ, hand_landmarks_xyZ = None, None

            if self._hand_detection_activation:
                hand_detector = self._left_camera_hand_detector if side == "left" else self._right_camera_hand_detector
                processed_img = np.copy(img)
                if self._mediapipe_cvt_color:
                    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=processed_img)

                hand_result = hand_detector.detect_for_video(mp_img, timestamp)
                hand_landmarks, handedness = get_normalized_hand_landmarks(hand_result)

            if self._body_detection_activation:
                body_detector = self._left_camera_body_detector if side == "left" else self._right_camera_body_detector 
                processed_img = np.copy(img)
                if self._mediapipe_cvt_color:
                    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=processed_img)

                body_detection_result = body_detector.detect_for_video(mp_img, timestamp)
                body_landmarks = get_normalized_pose_landmarks(body_detection_result)

            drawed_img = draw_arm_landmarks_on_image(drawed_img, body_landmarks)
            drawed_img = draw_hand_landmarks_on_image(drawed_img, hand_landmarks, handedness)

            if (hand_landmarks and 
                handedness and 
                self._hand_to_fuse in handedness):
                hand_id = handedness.index(self._hand_to_fuse)
                hand_landmarks = hand_landmarks[hand_id]

                hand_landmarks_xyZ = get_xyZ(hand_landmarks,  # shape: (21, 3) for single hand
                    img_size,
                    landmark_ids_to_get=None,
                    visibility_threshold=None,
                    depth_map=depth_map)

            if body_landmarks:
                if self._num_person_to_detect == 1:
                    # Get pose with highest area
                    body_landmarks = body_landmarks[0]
                
                body_landmarks_xyZ = get_xyZ(body_landmarks, 
                    img_size, 
                    self._body_landmarks_id_want_to_get,
                    self._body_visibility_threshold,
                    depth_map=depth_map)  # shape: (N, 3)

            if (body_landmarks_xyZ is not None and
                not np.isnan(body_landmarks_xyZ).any() and
                len(body_landmarks_xyZ) > 0 and
                hand_landmarks_xyZ is not None and
                not np.isnan(hand_landmarks_xyZ).any() and
                len(hand_landmarks_xyZ) > 0):
                body_hand_selected_xyZ = np.concatenate([body_landmarks_xyZ[:-1, :], # add right_elow after left_hand
                    hand_landmarks_xyZ,
                    body_landmarks_xyZ[-1, :][None, :]], axis=0)
        else:
            # -------------- MMPOSE DETECTION -------------- 
            detection_result = self._rtmpose_models(img, depth_map, timestamp, side)

        return detection_result