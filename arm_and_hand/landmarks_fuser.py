import torch
import numpy as np
from functools import partial

from transformer_encoder import TransformerEncoder
from utilities import (fuse_landmarks_from_two_cameras,
    convert_to_shoulder_coord,
    flatten_two_camera_input)

class LandmarksFuser:
    """
    Desc.
    """

    def __init__(self, method_selected_id, method_list,
        method_config, img_size):
        """
        Desc.

        Parameters:
            attribute1 (type): Description of attribute1.
            attribute2 (type): Description of attribute2.
        """
        self._method_name = method_list[method_selected_id]
        config_by_method = method_config[self._method_name]

        if self._method_name == "transformer_encoder":
            # ------------ INIT TRANSFORMER ENCODER ------------
            self._sequence_length = config_by_method["sequence_length"]
            input_dim = config_by_method["input_dim"]
            output_dim = config_by_method["output_dim"]
            num_heads = config_by_method["num_heads"]
            num_encoder_layers = config_by_method["num_encoder_layers"]
            dim_feedforward = config_by_method["dim_feedforward"]
            dropout = config_by_method["dropout"]
            model_path = config_by_method["model_weight_path"]
            self._input_sequence = []
            self._img_size = img_size

            self._fusing_model = TransformerEncoder(input_dim, 
                output_dim, 
                num_heads, 
                num_encoder_layers, 
                dim_feedforward, 
                dropout)
            self._fusing_model.load_state_dict(torch.load(model_path))
            self._fusing_model.to("cuda")
            self._fusing_model.eval()
        else:
            # ------------ INIT SCIPY OPTIMIZATION ------------
            self._min_distance_tol = float(config_by_method["tolerance"])
            self._min_distance_algo_name = config_by_method["algo_name"]
            self._fusing_model = partial(fuse_landmarks_from_two_cameras,
                tolerance = self._min_distance_tol,
                method_name = self._min_distance_algo_name)

    def fuse(self, left_camera_wholebody_xyZ, 
        right_camera_wholebody_xyZ,
        left_camera_intr,
        right_camera_intr,
        right_2_left_matrix):
        arm_hand_XYZ_wrt_shoulder, xyz_origin = None, None
        if self._method_name == "transformer_encoder":
            # -------------------- FUSE BY TRANSFORMER_ENCODER -------------------- 
            input_row = flatten_two_camera_input(left_camera_wholebody_xyZ,
                right_camera_wholebody_xyZ,
                left_camera_intr,
                right_camera_intr,
                right_2_left_matrix,
                self._img_size,
                mode="input")
            self._input_sequence.append(input_row) 

            if len(self._input_sequence) == self._sequence_length:
                x = np.array(self._input_sequence) 
                x = x[:, None, :]  # (seq, batch, num_inputs)
                x = torch.tensor(x, dtype=torch.float32)
                with torch.no_grad():
                    x = x.to("cuda")
                    y = self._fusing_model(x)
                    y = y.detach().to("cpu").numpy()[0]
                # FIX THIS ONE WHEN DONE MANUAL LABEL FOR FUSING MODEL
                # FOR NOW, THE MODEL SHOULD RETURN XYZ, MANUAL CONVERT
                # TO WRIST COORDINATE
                arm_hand_XYZ_wrt_shoulder = y[:144]
                xyz_origin = y[144:]
                arm_hand_XYZ_wrt_shoulder = arm_hand_XYZ_wrt_shoulder.reshape(3, 48)
                arm_hand_XYZ_wrt_shoulder = arm_hand_XYZ_wrt_shoulder.T
                xyz_origin = xyz_origin.reshape(3, 3)
                self._input_sequence = self._input_sequence[1:]
        else:
            # -------------------- FUSE BY OPTIMIZATION METHOD -------------------- 
            arm_hand_fused_XYZ = self._fusing_model(left_camera_wholebody_xyZ,
                right_camera_wholebody_xyZ,
                right_camera_intr,
                left_camera_intr,
                right_2_left_matrix)
            #arm_hand_XYZ_wrt_shoulder, xyz_origin = convert_to_shoulder_coord(
                #arm_hand_fused_XYZ,
                #self._arm_hand_fused_names)
        return arm_hand_fused_XYZ