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

class LandmarksDetectors:
    def __init__(self, model_selected_id, model_list, model_config, draw_landmarks):
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
        self._fusing_landmark_names = model_config["fusing_landmark_dictionary"]
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
            # -------------- INIT MMPOSE MODELS -------------- 
            self._mmpose_selected_landmarks_id = [
                5, 7, 11, 6, 12,  # thumb
                91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, # left hand
                8, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132  # right hand
                ]  
            self._mmpose_cvt_color = config_by_model["convert_color_channel"]
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

            if (self._draw_landmarks and 
                self._left_camera_pose_estimator is not None and
                self._right_camera_pose_estimator is not None):
                self._left_camera_pose_estimator.cfg.visualizer.line_width = 2
                self._right_camera_pose_estimator.cfg.visualizer.line_width = 2
                self._landmarks_thres = config_by_model["landmark_thresh"]

                # build the visualizer
                self._left_camera_mmpose_visualizer = VISUALIZERS.build(
                    self._left_camera_pose_estimator.cfg.visualizer)
                self._right_camera_mmpose_visualizer = VISUALIZERS.build(
                    self._right_camera_pose_estimator.cfg.visualizer)

                # set skeleton, colormap and joint connection rule
                self._left_camera_mmpose_visualizer.set_dataset_meta(
                    self._left_camera_pose_estimator.dataset_meta)
                self._right_camera_mmpose_visualizer.set_dataset_meta(
                    self._right_camera_pose_estimator.dataset_meta)
    
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
            person_detector, wholebody_detector = None, None
            bboxes = None
            if self._person_detection_activation:  
                person_detector = self._left_camera_person_detector if side == "left" else self._right_camera_person_detector
                processed_img = np.copy(img)
                if  self._mmpose_cvt_color:
                    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                det_result = inference_detector(person_detector, processed_img)
                pred_instance = det_result.pred_instances.cpu().numpy()
                bboxes = np.concatenate(
                    (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
                bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                    pred_instance.scores > 0.6)]
                bboxes = bboxes[nms(bboxes, 0.3), :]
                """Now we dont get a box with highest score, we should get box with 
                highest area"""
                if bboxes.size > 0:
                    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
                    largest_area_index = np.argmax(areas)
                    bboxes = bboxes[largest_area_index, :-1][None, :]
            if self._pose_estimation_activation:
                wholebody_detector = self._left_camera_pose_estimator if side == "left" else self._right_camera_pose_estimator
                if self._draw_landmarks:
                    visualizer = self._left_camera_mmpose_visualizer if side == "left" else self._right_camera_mmpose_visualizer
                processed_img = np.copy(img)
                if  self._mmpose_cvt_color:
                    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                if bboxes is not None and bboxes.size > 0: 
                    wholebody_det_rs = inference_topdown(wholebody_detector, processed_img, bboxes)
                else:
                    wholebody_det_rs = inference_topdown(wholebody_detector, processed_img)
                wholebody_det_rs = merge_data_samples(wholebody_det_rs)
                wholebody_preds = wholebody_det_rs.get("pred_instances", None)
                if wholebody_preds is not None:
                    wholebody_landmarks = wholebody_preds.keypoints[0]
                    wholebody_landmarks_score = wholebody_preds.keypoint_scores[0]
                    selected_score =  wholebody_landmarks_score[self._mmpose_selected_landmarks_id]
                    mask = selected_score > self._landmarks_thres
                    body_hand_selected_xyZ = wholebody_landmarks[self._mmpose_selected_landmarks_id]
                    body_hand_selected_xyZ = body_hand_selected_xyZ[mask]

                landmarks_depth = get_depth(body_hand_selected_xyZ, depth_map, 9)
                body_hand_selected_xyZ = np.concatenate([body_hand_selected_xyZ, landmarks_depth[:, None]], axis=-1)

                if self._draw_landmarks:
                    drawed_img = visualizer.add_datasample(
                        'result',
                        drawed_img,
                        data_sample=wholebody_det_rs,
                        draw_gt=False,
                        draw_heatmap=False,
                        draw_bbox=True,
                        show_kpt_idx=False,
                        skeleton_style="mmpose",
                        show=False,
                        wait_time=0,
                        kpt_thr=self._landmarks_thres
                    )

        detection_result = {"detection_result": body_hand_selected_xyZ,
            "drawed_img": drawed_img}

        return detection_result