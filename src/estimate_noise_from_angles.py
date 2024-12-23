"""
Estimate noise from angle when the arm or hand does not move. We will 
collect three values to put them into a Kalman Filter to reduce the noise:
    1. `init_angle`: The angle of each joint at the initial position.
    2. `cov`: The covariance matrix (or variance) of each joint at the initial position.
    3. `measure_noise`: Noise of each joint when the arm or hand in a fixed position.

In order to collect these three values, we must follow the following pipeline. Suppose 
that we want to reduce noise for left arm:
    1. Collect a bunch of left arm's landmarks in a fixed position. The first pose should
        be the initial pose of robot because we will use this single data file to 
        observer `init_angle` and `cov`. Each pose should be stored in a single file (.csv)
    2. Pass those data files (.csv) to a list. Note that the first file should be 
        a bunch of angles of a intial pose of robot. 
    3. Set the dimension of Kalman Filter. We can set `DIM = 6` (all joints are depend
        on each other) or `DIM = 1` (each joint is independent). In my experience, 
        set `DIM = 1` is the proper way. 
        
        *************************
        CURRENTLY, WE JUST SUPPORT FOR ARMS AND `DIM = 1`.
        *************************
"""

import os
import sys

sys.path.append("Hand_pose_estimation_3D/arm_and_hand")
sys.path.append("Hand_pose_estimation_3D")

import pandas as pd
import numpy as np
import json
from utilities import convert_to_left_shoulder_coord
from angle_calculator import (
    LeftArmAngleCalculator,
    RightArmAngleCalculator,
    LeftHandAngleCalculator
)

def get_angle_initial_expectation_and_cov(
    home_position_file, 
    arm_angle_calculator, 
    hand_angle_calculator,
    landmark_dictionary,
    side="left", 
    angles_from="arm"):
    home_position_data = pd.read_csv(home_position_csv_file)
    total_rows = len(home_position_data)

    # Calculate the starting and ending indices for the middle 50% data
    start_idx = total_rows // 4  # Start at 25% of the total rows
    end_idx = start_idx + total_rows // 2  # End at 75% of the total rows

    middle_data = home_position_data.iloc[start_idx:end_idx]
    gt_data = middle_data.loc[:, "left_shoulder_output_x":"right_pinky_tip_output_z"].values

    gt_data = gt_data.reshape(-1, 3, 48)
    gt_data = np.transpose(gt_data, (0, 2, 1))

    angles = []
    for i in range(gt_data.shape[0]):
        arm_hand_XYZ_wrt_shoulder, xyz_origin = convert_to_left_shoulder_coord(
            gt_data[i],
            landmark_dictionary) 

        arm_results = arm_angle_calculator(arm_hand_XYZ_wrt_shoulder, parent_coordinate=xyz_origin)
        if angles_from == "arm":
            angles_at_current_frame = arm_results[f"{side}_{angles_from}"]["angles"]
            angles_at_current_frame = np.array(angles_at_current_frame)
        elif angles_from == "hand":
            hand_angles = []

            parent_coord = arm_results[f"{side}_arm"]["rot_mats_wrt_origin"][-1]
            hand_results = hand_angle_calculator(arm_hand_XYZ_wrt_shoulder, parent_coord)
            
            for finger_name in hand_angle_calculator.fingers_name:
                finger_i_angles = hand_results[finger_name]["angles"]
                if finger_name != "THUMB":
                    finger_i_angles[0], finger_i_angles[1] = finger_i_angles[1], finger_i_angles[0]
                hand_angles.extend(finger_i_angles)
            angles_at_current_frame = np.array(hand_angles)
        
        angles.append(angles_at_current_frame)

    angles = np.array(angles)
    home_init_expectation = np.mean(angles, axis=0)
    home_init_cov = np.std(angles, axis=0)

    return home_init_expectation, home_init_cov

def calculate_noise_of_fixed_pose(
    measure_noise_csv_files, 
    arm_angle_calculator, 
    hand_angle_calculator,
    landmark_dictionary,
    side="left", 
    angles_from="arm"):
    angles_var = []

    for csv_file in measure_noise_csv_files:
        data = pd.read_csv(csv_file)
        total_rows = len(data)

        off_set = 10
        start_idx = total_rows // 4  
        end_idx = start_idx + total_rows // 2  

        middle_data = data.iloc[max(0, start_idx - off_set):end_idx]
        gt_data = middle_data.loc[:, "left_shoulder_output_x":"right_pinky_tip_output_z"].values
        gt_data = gt_data.reshape(-1, 3, 48)
        gt_data = np.transpose(gt_data, (0, 2, 1))

        angles_through_frame = []
        for i in range(gt_data.shape[0]):
            arm_hand_XYZ_wrt_shoulder, xyz_origin = convert_to_left_shoulder_coord(
                gt_data[i],
                landmark_dictionary)

            arm_result = arm_angle_calculator(arm_hand_XYZ_wrt_shoulder,
                xyz_origin)
            if angles_from == "arm":
                angles_at_current_frame = arm_result[f"{side}_{angles_from}"]["angles"]
                angles_at_current_frame = np.array(angles_at_current_frame)
            elif angles_from == "hand":
                hand_angles = []

                parent_coord = arm_result[f"{side}_arm"]["rot_mats_wrt_origin"][-1]
                hand_results = hand_angle_calculator(arm_hand_XYZ_wrt_shoulder, parent_coord)
            
                for finger_name in hand_angle_calculator.fingers_name:
                    finger_i_angles = hand_results[finger_name]["angles"]
                    if finger_name != "THUMB":
                        finger_i_angles[0], finger_i_angles[1] = finger_i_angles[1], finger_i_angles[0]
                    hand_angles.extend(finger_i_angles)
                angles_at_current_frame = np.array(hand_angles)

            angles_through_frame.append(angles_at_current_frame)

        angles_through_frame = np.array(angles_through_frame)
        angles_var.append(np.std(angles_through_frame, axis=0))
    
    angles_var = np.array(angles_var)
    noise_measured = np.mean(angles_var, axis=0)
    return noise_measured

if __name__ == "__main__":
    measure_noise_csv_files = [
        "data/2024-10-19/2024-10-19-15:34/landmarks_all_2024-10-19-15:34.csv",
        "data/2024-10-19/2024-10-19-15:37/landmarks_all_2024-10-19-15:37.csv",
        "data/2024-10-19/2024-10-19-15:38/landmarks_all_2024-10-19-15:38.csv",
        "data/2024-10-19/2024-10-19-15:39/landmarks_all_2024-10-19-15:39.csv",
        "data/2024-10-19/2024-10-19-15:40/landmarks_all_2024-10-19-15:40.csv",
        "data/2024-10-19/2024-10-19-15:41/landmarks_all_2024-10-19-15:41.csv"
    ]
    
    landmark_dictionary = ["left shoulder", "left elbow", "left hip", "right shoulder",
     "right hip", "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", 
     "THUMB_TIP", "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP",
     "INDEX_FINGER_TIP", "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP",
     "MIDDLE_FINGER_TIP", "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP",
     "RING_FINGER_TIP", "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP", "right elbow",
     "RIGHT_WRIST", "RIGHT_THUMB_CMC", "RIGHT_THUMB_MCP", "RIGHT_THUMB_IP", "RIGHT_THUMB_TIP",
     "RIGHT_INDEX_FINGER_MCP", "RIGHT_INDEX_FINGER_PIP", "RIGHT_INDEX_FINGER_DIP",
     "RIGHT_INDEX_FINGER_TIP", "RIGHT_MIDDLE_FINGER_MCP", "RIGHT_MIDDLE_FINGER_PIP",
     "RIGHT_MIDDLE_FINGER_DIP", "RIGHT_MIDDLE_FINGER_TIP", "RIGHT_RING_FINGER_MCP",
     "RIGHT_RING_FINGER_PIP", "RIGHT_RING_FINGER_DIP", "RIGHT_RING_FINGER_TIP",
     "RIGHT_PINKY_MCP", "RIGHT_PINKY_PIP", "RIGHT_PINKY_DIP", "RIGHT_PINKY_TIP"]
    
    SIDE = "left" 
    assert SIDE in ["left", "right"]
    
    ESTIMATE_NOISE_FOR = "hand" 
    assert ESTIMATE_NOISE_FOR in ["arm", "hand"]

    if ESTIMATE_NOISE_FOR == "arm":
        NUM_ANGLES = 6 
    elif ESTIMATE_NOISE_FOR == "hand":
        NUM_ANGLES = 15

    DIM = 1
    SAVE_DIR = "Hand_pose_estimation_3D/arm_and_hand/configuration"
    FILE_NAME = f"{SIDE}_{ESTIMATE_NOISE_FOR}_angles_stats.json"
    DES_JSON_FILE = os.path.join(SAVE_DIR, FILE_NAME)
    
    if SIDE == "left": 
        arm_calculator = LeftArmAngleCalculator(3, landmark_dictionary)
        hand_calculator = LeftHandAngleCalculator(2, landmark_dictionary)
    if SIDE == "right":
        if ESTIMATE_NOISE_FOR == "arm":
            angle_calculator = RightArmAngleCalculator(3, landmark_dictionary)
        else:
            #TODO: set for right hand angle calculation
            pass

    home_position_csv_file = measure_noise_csv_files[0]
    home_init_expectation, home_init_cov = get_angle_initial_expectation_and_cov(
        home_position_csv_file,
        arm_calculator,
        hand_calculator,
        landmark_dictionary,
        side=SIDE,
        angles_from=ESTIMATE_NOISE_FOR)
    angles_measure_noises = calculate_noise_of_fixed_pose(
        measure_noise_csv_files,
        arm_calculator,
        hand_calculator,
        landmark_dictionary,
        side=SIDE,
        angles_from=ESTIMATE_NOISE_FOR)

    angles_stats_dict = dict()

    if ESTIMATE_NOISE_FOR == "arm":
        for i in range(NUM_ANGLES):
            stats = dict()
            angles_measure_noise = angles_measure_noises[i]
            init_exp = home_init_expectation[i]
            init_cov = home_init_cov[i]

            stats["measure_noise"] = angles_measure_noise
            stats["init_angle"] = init_exp
            stats["cov"] = init_cov

            angles_stats_dict["joint{}".format(i + 1)] = stats
    elif ESTIMATE_NOISE_FOR == "hand":
        FINGERS_NAME = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]
        NUM_ANGLES_EACH_FINGER = 3
        assert NUM_ANGLES_EACH_FINGER * len(FINGERS_NAME) == NUM_ANGLES
        for angle_idx in range(NUM_ANGLES):
            finger_i = angle_idx // NUM_ANGLES_EACH_FINGER
            joint_of_finger_i = angle_idx % NUM_ANGLES_EACH_FINGER

            if joint_of_finger_i == 0:
                angles_stats_dict[FINGERS_NAME[finger_i]] = dict()

            stats = dict()

            angles_measure_noise = angles_measure_noises[angle_idx]
            init_exp = home_init_expectation[angle_idx]
            init_cov = home_init_cov[angle_idx]

            stats["measure_noise"] = angles_measure_noise,
            stats["init_angle"] = init_exp
            stats["cov"] = init_cov
            
            angles_stats_dict[FINGERS_NAME[finger_i]][f"joint{joint_of_finger_i + 1}"] = stats
        
    with open(DES_JSON_FILE, 'w') as file:
        json.dump(angles_stats_dict, file)
        