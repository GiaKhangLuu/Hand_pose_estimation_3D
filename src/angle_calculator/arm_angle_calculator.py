import numpy as np
from scipy.spatial.transform import Rotation as R
from .chain_angle_calculator import ChainAngleCalculator

class ArmAngleCalculator(ChainAngleCalculator):
    def __init__(self, num_chain, landmark_dictionary, side):
        self.side = side
        super().__init__(num_chain, landmark_dictionary)

    def __call__(self, XYZ_landmarks, parent_coordinate):
        merged_result_dict = dict()
        chain_result_dict = self._calculate_chain_angles(XYZ_landmarks, parent_coordinate)
        arm_angles = []
        rot_mats_wrt_origin = []
        rot_mats_wrt_parent = []

        for chain_idx in range(self.num_chain):
            chain_result = chain_result_dict[f"chain_{chain_idx+1}"] 

            older_brother_angle = chain_result["older_brother_angle"]
            older_brother_rot_mat_wrt_origin = chain_result["older_brother_rot_mat_wrt_origin"]
            older_brother_rot_mat_wrt_parent = chain_result["older_brother_rot_mat_wrt_parent"]
            younger_brother_angle = chain_result["younger_brother_angle"]
            younger_brother_rot_mat_wrt_origin = chain_result["younger_brother_rot_mat_wrt_origin"]
            younger_brother_rot_mat_wrt_older_brother = chain_result["younger_brother_rot_mat_wrt_older_brother"]
            vector_in_current_frame = chain_result["vector_in_current_frame"].copy()

            self._update_vector_in_previous_frame(chain_idx, vector_in_current_frame)

            arm_angles.extend([older_brother_angle, younger_brother_angle])
            rot_mats_wrt_origin.extend([older_brother_rot_mat_wrt_origin, younger_brother_rot_mat_wrt_origin])
            rot_mats_wrt_parent.extend([older_brother_rot_mat_wrt_parent, younger_brother_rot_mat_wrt_older_brother])

        merged_result_dict[f"{self.side}_arm"] = {
            "angles": arm_angles,
            "rot_mats_wrt_origin": rot_mats_wrt_origin,
            "rot_mats_wrt_parent": rot_mats_wrt_parent
        }

        return merged_result_dict