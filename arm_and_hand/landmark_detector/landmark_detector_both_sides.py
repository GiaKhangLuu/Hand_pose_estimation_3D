import os
from pathlib import Path
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

WORK_DIR = Path(os.environ["MOCAP_WORKDIR"])

class RTMPoseDetector_BothSides:
    def __init__(self, model_config, draw_landmarks):
        self._draw_landmarks = draw_landmarks
        self._mmpose_selected_landmarks_id = list(model_config["selected_landmarks_idx"])
        self._mmpose_cvt_color = model_config["convert_color_channel"]
        self._person_detection_activation = model_config["person_detection"]["is_enable"]
        self._landmarks_thres = model_config["landmark_thresh"]
        person_detector_config = str(WORK_DIR / model_config["person_detection"]["person_detector_config"])
        person_detector_weight = str(WORK_DIR / model_config["person_detection"]["person_detector_weight"])
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
        pose_estimator_config = str(WORK_DIR / model_config["pose_estimation"]["pose_estimator_config"])
        pose_estimator_weight = str(WORK_DIR / model_config["pose_estimation"]["pose_estimator_weight"])
        if self._pose_estimation_activation:
            # pose_estimator for left camera
            if os.path.isdir(pose_estimator_weight):
                # Run TensorRT model
                self._pose_estimator = PoseDetector(
                    model_path=pose_estimator_weight,
                    device_name="cuda",
                    device_id=0)
            else:
                # Run Pytorch model
                self._pose_estimator = init_pose_estimator(
                    pose_estimator_config,
                    pose_estimator_weight,
                    device="cuda")

        if self._draw_landmarks:
            if self._pose_estimation_activation:
                pose_config = Config.fromfile(pose_estimator_config)
                pose_config.visualizer.line_width = 2
                self._visualizer = VISUALIZERS.build(pose_config.visualizer)
                dataset_meta = dataset_meta_from_config(pose_config, dataset_mode="train")
                if dataset_meta is None:
                    dataset_meta = parse_pose_metainfo(
                        dict(from_file='configs/_base_/datasets/coco.py'))
                self._visualizer.set_dataset_meta(dataset_meta)

    def __call__(self, detection_input):
        # Currently, support TensorRT runtime only
        left_cam_rgb = detection_input["left_cam_rgb"]
        right_cam_rgb = detection_input["right_cam_rgb"]
        left_cam_depth = detection_input["left_cam_depth"]
        right_cam_depth = detection_input["right_cam_depth"]
        depth_container = [left_cam_depth, right_cam_depth]
        person_detected_bboxes = [None, None]

        selected_landmarks_container = [None, None]
        drawed_img_container = [np.copy(left_cam_rgb), np.copy(right_cam_rgb)]
        
        pose_detection_rs = {
            "left_cam_detection_result": None,
            "right_cam_detection_result": None,
            "left_cam_drawed_img": None,
            "right_cam_drawed_img": None,
        }

        if self._person_detection_activation:
            processed_left_img = np.copy(left_cam_rgb)
            processed_right_img = np.copy(right_cam_rgb)
            if self._mmpose_cvt_color:
                processed_left_img = cv2.cvtColor(processed_left_img, cv2.COLOR_BGR2RGB)
                processed_right_img = cv2.cvtColor(processed_right_img, cv2.COLOR_BGR2RGB)
            batch_img = [processed_left_img, processed_right_img]
            detection_rs = self._person_detector.batch(batch_img)

            batch_bboxes_det = [rs[0] for rs in detection_rs]
            batch_labels_det = [rs[1] for rs in detection_rs]

            for i, (bboxes, labels) in enumerate(zip(batch_bboxes_det, batch_labels_det)):
                bboxes = bboxes[np.logical_and(labels == 0, bboxes[..., -1] > 0.6)]

                if bboxes.shape[0] > 0:
                    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
                    largest_area_index = np.argmax(areas)
                    bboxes = bboxes[largest_area_index, :-1][None, :]
                    person_detected_bboxes[i] = bboxes 

        if self._pose_estimation_activation:
            processed_left_img = np.copy(left_cam_rgb)
            processed_right_img = np.copy(right_cam_rgb)
            if self._mmpose_cvt_color:
                processed_left_img = cv2.cvtColor(processed_left_img, cv2.COLOR_BGR2RGB)
                processed_right_img = cv2.cvtColor(processed_right_img, cv2.COLOR_BGR2RGB)
            batch_img = [processed_left_img, processed_right_img]
            if (person_detected_bboxes[0] is not None and 
                person_detected_bboxes[1] is not None): 
                wholebody_det_rs_batch = self._pose_estimator.batch(batch_img, person_detected_bboxes)
            else:
                wholebody_det_rs_batch = self._pose_estimator.batch(batch_img)

            if wholebody_det_rs_batch is not None:
                for i, wholebody_det_rs in enumerate(wholebody_det_rs_batch):
                    scores = wholebody_det_rs[0, :, 2]
                    keypoints = wholebody_det_rs[0, :, :2]
                    selected_score = scores[self._mmpose_selected_landmarks_id]
                    mask = selected_score > self._landmarks_thres
                    body_hand_selected_xyZ = keypoints[self._mmpose_selected_landmarks_id]
                    body_hand_selected_xyZ = body_hand_selected_xyZ[mask]

                    depth_map = depth_container[i]
                    landmarks_depth = get_depth(body_hand_selected_xyZ, depth_map, 9)
                    body_hand_selected_xyZ = np.concatenate([body_hand_selected_xyZ, landmarks_depth[:, None]], axis=-1)

                    selected_landmarks_container[i] = body_hand_selected_xyZ
            
                    if self._draw_landmarks:
                        pred_instances = InstanceData()
                        pred_instances.keypoints = wholebody_det_rs[0, :, :2][None, :]
                        pred_instances.score = wholebody_det_rs[0, :, 2][None, :]
                        if (self._person_detection_activation and 
                            person_detected_bboxes[i] is not None and
                            person_detected_bboxes[i].shape[0] > 0):
                            pred_instances.bboxes = person_detected_bboxes[i]
                        pred_pose_data_sample = PoseDataSample()
                        pred_pose_data_sample.pred_instances = pred_instances
                        drawed_img = self._visualizer.add_datasample(
                            "image",
                            drawed_img_container[i],
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

                        drawed_img_container[i] = drawed_img

        pose_detection_rs["left_cam_detection_result"] = selected_landmarks_container[0]
        pose_detection_rs["right_cam_detection_result"] = selected_landmarks_container[1]
        pose_detection_rs["left_cam_drawed_img"] = drawed_img_container[0]
        pose_detection_rs["right_cam_drawed_img"] = drawed_img_container[1]

        return pose_detection_rs