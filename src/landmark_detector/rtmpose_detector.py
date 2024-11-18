import os
import numpy as np
#import warning
from mmdeploy_runtime import Detector, PoseDetector

from mmcv.image import imread
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import init_model as init_pose_estimator
from mmpose.apis import inference_topdown
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from mmpose.evaluation.functional import nms
from mmpose.utils import adapt_mmdet_pipeline
from mmengine.config import Config
from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.apis.inference import dataset_meta_from_config
from mmpose.datasets.datasets.utils import parse_pose_metainfo

from mmengine.structures import InstanceData
from mmpose.structures import PoseDataSample

from common import (
    get_xyZ,
    get_depth
)

class RTMPoseDetector:
    def __init__(self, model_config, draw_landmarks):
        self._draw_landmarks = draw_landmarks
        self._mmpose_selected_landmarks_id = list(model_config["selected_landmarks_idx"])
        self._mmpose_cvt_color = model_config["convert_color_channel"]
        self._person_detection_activation = model_config["person_detection"]["is_enable"]
        self._landmarks_thres = model_config["landmark_thresh"]
        person_detector_config = model_config["person_detection"]["person_detector_config"]
        person_detector_weight = model_config["person_detection"]["person_detector_weight"]
        if self._person_detection_activation:
            if os.path.isdir(person_detector_weight):
                # Run TensorRT model
                self._person_detector = Detector(
                    model_path=person_detector_weight,
                    device_name="cuda", 
                    device_id=0)
            else:
                # Run Pytorch model
                self._person_detector = init_detector(
                    person_detector_config, 
                    person_detector_weight,
                    device="cuda")
                self._person_detector.cfg = adapt_mmdet_pipeline(self._person_detector.cfg)
        else:
            self._person_detector = None

        self._pose_estimation_activation = model_config["pose_estimation"]["is_enable"]
        left_camera_pose_estimator_config = model_config["pose_estimation"]["left_camera_pose_estimator_config"]
        left_camera_pose_estimator_weight = model_config["pose_estimation"]["left_camera_pose_estimator_weight"]
        right_camera_pose_estimator_config = model_config["pose_estimation"]["right_camera_pose_estimator_config"]
        right_camera_pose_estimator_weight = model_config["pose_estimation"]["right_camera_pose_estimator_weight"]
        if self._pose_estimation_activation:
            # pose_estimator for left camera
            if os.path.isdir(left_camera_pose_estimator_weight):
                # Run TensorRT model
                self._left_camera_pose_estimator = PoseDetector(
                    model_path=left_camera_pose_estimator_weight,
                    device_name="cuda",
                    device_id=0)
            else:
                # Run Pytorch model
                self._left_camera_pose_estimator = init_pose_estimator(
                    left_camera_pose_estimator_config,
                    left_camera_pose_estimator_weight,
                    device="cuda")

            # pose_estimator for right camera
            if os.path.isdir(right_camera_pose_estimator_weight):
                # Run TensorRT model
                self._right_camera_pose_estimator = PoseDetector(
                    model_path=right_camera_pose_estimator_weight,
                    device_name="cuda",
                    device_id=0)
            else:
                # Run Pytorch model
                self._right_camera_pose_estimator = init_pose_estimator(
                    right_camera_pose_estimator_config,
                    right_camera_pose_estimator_weight,
                    device_id=0)

        if self._draw_landmarks:
            if self._pose_estimation_activation:
                left_camera_config = Config.fromfile(left_camera_pose_estimator_config)
                left_camera_config.visualizer.line_width = 2
                self._left_camera_mmpose_visualizer = VISUALIZERS.build(left_camera_config.visualizer)
                dataset_meta = dataset_meta_from_config(left_camera_config, dataset_mode="train")
                if dataset_meta is None:
                    #warnings.simplefilter('once')
                    #warnings.warn('Can not load dataset_meta from the checkpoint or the '
                                #'model config. Use COCO metainfo by default.')
                    dataset_meta = parse_pose_metainfo(
                        dict(from_file='configs/_base_/datasets/coco.py'))
                self._left_camera_mmpose_visualizer.set_dataset_meta(dataset_meta)

                right_camera_config = Config.fromfile(right_camera_pose_estimator_config)
                right_camera_config.visualizer.line_width = 2
                self._right_camera_mmpose_visualizer = VISUALIZERS.build(right_camera_config.visualizer)
                dataset_meta = dataset_meta_from_config(right_camera_config, dataset_mode="train")
                if dataset_meta is None:
                    #warnings.simplefilter('once')
                    #warnings.warn('Can not load dataset_meta from the checkpoint or the '
                                #'model config. Use COCO metainfo by default.')
                    dataset_meta = parse_pose_metainfo(
                        dict(from_file='configs/_base_/datasets/coco.py'))
                self._right_camera_mmpose_visualizer.set_dataset_meta(dataset_meta)

    def __call__(self, img, depth_map, timestamp, side):
        # Currently, support TensorRT runtime only
        bboxes = None
        body_hand_selected_xyZ = None
        drawed_img = np.copy(img)
        if self._person_detection_activation:
            processed_img = np.copy(img)
            if self._mmpose_cvt_color:
                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            bboxes, labels, _ = self._person_detector(processed_img)
            bboxes = bboxes[np.logical_and(labels == 0,
                bboxes[..., 4] > 0.6)]

            if bboxes.shape[0] > 0:
                areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
                largest_area_index = np.argmax(areas)
                bboxes = bboxes[largest_area_index, :-1][None, :]

        if self._pose_estimation_activation:
            wholebody_detector = self._left_camera_pose_estimator if side == "left" else self._right_camera_pose_estimator
            processed_img = np.copy(img)
            if  self._mmpose_cvt_color:
                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            if bboxes is not None and bboxes.shape[0] > 0: 
                wholebody_det_rs = wholebody_detector(processed_img, bboxes)
            else:
                wholebody_det_rs = wholebody_detector(processed_img)
            if wholebody_det_rs is not None:
                scores = wholebody_det_rs[0, :, 2]
                keypoints = wholebody_det_rs[0, :, :2]
                selected_score = scores[self._mmpose_selected_landmarks_id]
                mask = selected_score > self._landmarks_thres
                body_hand_selected_xyZ = keypoints[self._mmpose_selected_landmarks_id]
                body_hand_selected_xyZ = body_hand_selected_xyZ[mask]

                landmarks_depth = get_depth(body_hand_selected_xyZ, depth_map, 9)
                body_hand_selected_xyZ = np.concatenate([body_hand_selected_xyZ, landmarks_depth[:, None]], axis=-1)
            
            # TODO: Draw image
            if self._draw_landmarks:
                visualizer = self._left_camera_mmpose_visualizer if side == "left" else self._right_camera_mmpose_visualizer
                pred_instances = InstanceData()
                pred_instances.keypoints = wholebody_det_rs[0, :, :2][None, :]
                pred_instances.score = wholebody_det_rs[0, :, 2][None, :]
                if self._person_detection_activation and bboxes.shape[0] > 0:
                    pred_instances.bboxes = bboxes
                pred_pose_data_sample = PoseDataSample()
                pred_pose_data_sample.pred_instances = pred_instances
                drawed_img = visualizer.add_datasample(
                    "image",
                    drawed_img,
                    data_sample=pred_pose_data_sample,
                    draw_gt=False,
                    draw_heatmap=False,
                    draw_bbox=True,
                    show_kpt_idx=False,
                    skeleton_style="mmpose",
                    show=False,
                    wait_time=0,
                    kpt_thr=self._landmarks_thres
                )

        detection_result = {
            "detection_result": body_hand_selected_xyZ,
            "drawed_img": drawed_img
        }

        return detection_result