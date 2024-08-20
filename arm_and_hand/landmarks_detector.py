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

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

from utilities import (get_normalized_hand_landmarks,
    get_normalized_pose_landmarks,
    get_landmarks_name_based_on_arm,
    get_mediapipe_world_landmarks,
    xyZ_to_XYZ)
from common import get_xyZ

class LandmarksDetectors:
    def __init__(self, model_selected_id, model_list, model_config):
        """
        Desc.
        Parameters:
            attribute1 (type): Description of attribute1.
            attribute2 (type): Description of attribute2.
        """

        self._model_name = model_list[model_selected_id]
        config_by_model = model_config[self._model_name]
        self._frame_color_format = model_config["frame_color_format"]
        hand_to_fuse = model_config["hand_to_fuse"]
        arm_to_fuse = model_config["arm_to_fuse"]
        self._fusing_landmark_names = model_config["fusing_landmark_dictionary"]
        self._num_person_to_detect = model_config["num_person_to_detect"]

        assert hand_to_fuse in ["left", "right", "both"]
        assert arm_to_fuse == hand_to_fuse

        if self._model_name == "mediapipe":
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
            body_landmarks_name = tuple(config_by_model["body_detection"]["pose_landmarks"])

            self._hand_to_fuse = "Left" if hand_to_fuse else "Right"  # Currently select left or right (fix this if get both left and right)
            body_landmarks_names_want_to_get = get_landmarks_name_based_on_arm(arm_to_fuse)
            self._body_landmarks_id_want_to_get = [body_landmarks_name.index(name) for name in body_landmarks_names_want_to_get]
            self._body_hand_fused_names = body_landmarks_names_want_to_get.copy()
            self._body_hand_fused_names.extend(hand_landmarks_name)

            assert self._body_hand_fused_names == self._fusing_landmark_names

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
            self._person_detection_activation = config_by_model["person_detection"]["is_enable"]
            person_detector_config = config_by_model["person_detection"]["person_detector_config"]
            person_detector_weight = config_by_model["person_detection"]["person_detector_weight"]
            if self._person_detection_activation:
                self._left_camera_person_detector = init_detector(person_detector_config,
                    person_detector_weight, device="cuda")
                self._left_camera_person_detector.cfg = adapt_mmdet_pipeline(self._left_camera_person_detector.cfg)
                self._right_camera_person_detector = init_detector(person_detector_config,
                    person_detector_weight, device="cuda")
                self._right_camera_person_detector.cfg = adapt_mmdet_pipeline(self._right_camera_person_detector.cfg)
            else:
                self._left_camera_person_detector = None
                self._right_camera_person_detector = None

            self._pose_estimation_activation = config_by_model["pose_estimation"]["is_enable"]
            left_camera_pose_estimator_config = config_by_model["pose_estimation"]["left_camera_pose_estimator_config"]
            left_camera_pose_estimator_weight = config_by_model["pose_estimation"]["left_camera_pose_estimator_weight"]
            right_camera_pose_estimator_config = config_by_model["pose_estimation"]["right_camera_pose_estimator_config"]
            right_camera_pose_estimator_weight = config_by_model["pose_estimation"]["right_camera_pose_estimator_weight"]
            if self._pose_estimation_activation:
                self._left_camera_pose_estimator = init_pose_estimator(left_camera_pose_estimator_config, 
                    left_camera_pose_estimator_weight, device="cuda") 
                self._right_camera_pose_estimator = init_pose_estimator(right_camera_pose_estimator_config, 
                    right_camera_pose_estimator_weight, device="cuda") 

    def detect(self, img, depth_map, side, timestamp):
        assert side in ["left", "right"]
        img_size = img.shape[:-1]
        body_hand_selected_xyZ = None
        if self._model_name == "mediapipe":
            hand_detector, body_detector = None, None
            if self._hand_detection_activation:
                processed_img = np.copy(img)
                if frame_color_format == "bgr":
                    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=processed_img)

                hand_result = hand_detector.detect_for_video(mp_img, timestamp)
                hand_landmarks, handedness = get_normalized_hand_landmarks(hand_result)

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

            if self._body_detection_activation:
                body_detector = self._left_camera_body_detector if side == "left" else self._right_camera_body_detector 
                processed_img = np.copy(img)
                if self._frame_color_format == "bgr":
                    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=processed_img)

                body_detection_result = body_detector.detect_for_video(mp_img, timestamp)
                body_landmarks = get_normalized_pose_landmarks(body_detection_result)

                if body_landmarks:
                    if self._num_person_to_detect == 1:
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
                body_hand_selected_xyZ = np.concatenate([body_landmarks_xyZ, hand_landmarks_xyZ], axis=0)
        else:
            if self._person_detection_activation:  
                det_result = inference_detector(left_detector, left_rgb)
                left_pred_instance = left_det_result.pred_instances.cpu().numpy()
                left_bboxes = np.concatenate(
                    (left_pred_instance.bboxes, left_pred_instance.scores[:, None]), axis=1)
                left_bboxes = left_bboxes[np.logical_and(left_pred_instance.labels == 0,
                                            left_pred_instance.scores > 0.8)]
                left_bboxes = left_bboxes[nms(left_bboxes, 0.5), :4]
            
            
        return body_hand_selected_xyZ


        left_det_result = inference_detector(left_detector, left_rgb)
        left_pred_instance = left_det_result.pred_instances.cpu().numpy()
        left_bboxes = np.concatenate(
            (left_pred_instance.bboxes, left_pred_instance.scores[:, None]), axis=1)
        left_bboxes = left_bboxes[np.logical_and(left_pred_instance.labels == 0,
                                    left_pred_instance.scores > 0.8)]
        left_bboxes = left_bboxes[nms(left_bboxes, 0.5), :4]

        right_det_result = inference_detector(right_detector, right_rgb)
        right_pred_instance = right_det_result.pred_instances.cpu().numpy()
        right_bboxes = np.concatenate(
            (right_pred_instance.bboxes, right_pred_instance.scores[:, None]), axis=1)
        right_bboxes = right_bboxes[np.logical_and(right_pred_instance.labels == 0,
                                    right_pred_instance.scores > 0.8)]
        right_bboxes = right_bboxes[nms(right_bboxes, 0.5), :4]

        left_detection_result = inference_topdown(left_cam_pose_estimator, left_rgb, left_bboxes)
        right_detection_result = inference_topdown(right_cam_pose_estimator, right_rgb, right_bboxes)

        left_detection_result = merge_data_samples(left_detection_result)
        right_detection_result = merge_data_samples(right_detection_result)

        left_rgb = left_cam_visualizer.add_datasample(
            'opposite_result',
            left_rgb,
            data_sample=left_detection_result,
            draw_gt=False,
            draw_heatmap=False,
            draw_bbox=True,
            show_kpt_idx=False,
            skeleton_style="mmpose",
            show=False,
            wait_time=0,
            kpt_thr=0.3
        )

        right_rgb = right_cam_visualizer.add_datasample(
            'rightside_result',
            right_rgb,
            data_sample=right_detection_result,
            draw_gt=False,
            draw_heatmap=False,
            draw_bbox=True,
            show_kpt_idx=False,
            skeleton_style="mmpose",
            show=False,
            wait_time=0,
            kpt_thr=0.3
        )

        left_predictions = left_detection_result.get("pred_instances", None)
        right_predictions = right_detection_result.get("pred_instances", None)

        if left_predictions is not None and right_predictions is not None:
            left_body_landmarks = left_predictions.keypoints[0]
            right_body_landmarks = right_predictions.keypoints[0]

            landmarks_id_want_to_get = [5, 7, 11, 6, 12, 91, 92, 93, 94, 95, 
                96, 97, 98, 99, 100, 101, 102, 103, 104, 
                105, 106, 107, 108, 109, 110, 111]

            left_selected_landmarks = left_body_landmarks[landmarks_id_want_to_get]
            right_selected_landmarks = right_body_landmarks[landmarks_id_want_to_get]

            left_Z = get_depth(left_selected_landmarks, left_depth, 9)
            left_selected_landmarks_xyZ = np.concatenate([left_selected_landmarks, left_Z[:, None]], axis=-1)
            right_Z = get_depth(right_selected_landmarks, right_depth, 9)
            right_selected_landmarks_xyZ = np.concatenate([right_selected_landmarks, right_Z[:, None]], axis=-1)




