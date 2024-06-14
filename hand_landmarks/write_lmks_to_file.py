import sys
import numpy as np
import os
import time
from datetime import datetime
from sklearn.model_selection import train_test_split

CURR_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(CURR_FOLDER, "data")

sys.path.append(os.path.join(CURR_FOLDER, '..'))

from hand_landmarks.tools import xyZ_to_XYZ, transform_to_another_coordinate

input_cam1_cols = ["cam1_X00_in", "cam1_Y00_in", "cam1_Z00_in",
                   "cam1_X01_in", "cam1_Y01_in", "cam1_Z01_in",
                   "cam1_X02_in", "cam1_Y02_in", "cam1_Z02_in",
                   "cam1_X03_in", "cam1_Y03_in", "cam1_Z03_in",
                   "cam1_X04_in", "cam1_Y04_in", "cam1_Z04_in",
                   "cam1_X05_in", "cam1_Y05_in", "cam1_Z05_in",
                   "cam1_X06_in", "cam1_Y06_in", "cam1_Z06_in",
                   "cam1_X07_in", "cam1_Y07_in", "cam1_Z07_in",
                   "cam1_X08_in", "cam1_Y08_in", "cam1_Z08_in",
                   "cam1_X09_in", "cam1_Y09_in", "cam1_Z09_in",
                   "cam1_X10_in", "cam1_Y10_in", "cam1_Z10_in",
                   "cam1_X11_in", "cam1_Y11_in", "cam1_Z11_in",
                   "cam1_X12_in", "cam1_Y12_in", "cam1_Z12_in",
                   "cam1_X13_in", "cam1_Y13_in", "cam1_Z13_in",
                   "cam1_X14_in", "cam1_Y14_in", "cam1_Z14_in",
                   "cam1_X15_in", "cam1_Y15_in", "cam1_Z15_in",
                   "cam1_X16_in", "cam1_Y16_in", "cam1_Z16_in",
                   "cam1_X17_in", "cam1_Y17_in", "cam1_Z17_in",
                   "cam1_X18_in", "cam1_Y18_in", "cam1_Z18_in",
                   "cam1_X19_in", "cam1_Y19_in", "cam1_Z19_in",
                   "cam1_X20_in", "cam1_Y20_in", "cam1_Z20_in"]

input_cam1_header = ','.join(input_cam1_cols)

def write_data_to_csv(file_path, data, num_cam=2):
    num_points_each_joint = 3
    num_joints_each_hand = 21
    num_input_cols = num_cam * num_points_each_joint * num_joints_each_hand

    input_header = input_cam1_header
    for i in range(2, num_cam+1):
        input_cam_i_header = input_cam1_header.replace("cam1", "cam{}".format(i))
        input_header += ',' + input_cam_i_header

    output_header = input_cam1_header.replace("cam1_", "").replace("in", "out")
    csv_header = input_header + ',' + output_header

    assert len(csv_header.split(",")) == data.shape[1]

    np.savetxt(file_path, data, delimiter=',', fmt='%f', header=csv_header, comments='')

def write_lnmks_to_file(lmks_queue, rs_ins, oak_ins, oak_2_rs_mat):
    count = 0

    xyZ_of_opposite_cam_through_frames = []
    xyZ_of_rightside_cam_through_frames = []
    lmks_gt_through_frames = []

    # -------------------- WARM UP -------------------- 
    time.sleep(20)

    while count >= 0:

        if not lmks_queue.empty():
            print('----------------------- Storing hand landmarks ----------------------- ')

            # raw_xyZ_of_opposite_cam: (21, 3)
            # raw_xyZ_of_right_side_cam: (21, 3)                                
            # wrist_gt: (3,)
            # finger_lmks_gt: (5, 4, 3)
            landmarks = lmks_queue.get() 
            raw_xyZ_of_opposite_cam = landmarks[0]
            raw_xyZ_of_right_side_cam = landmarks[1]
            wrist_gt = landmarks[2]
            finger_lmks_gt =landmarks[3]

            # Store landmarks input
            xyZ_of_opposite_cam_through_frames.append(raw_xyZ_of_opposite_cam)
            xyZ_of_rightside_cam_through_frames.append(raw_xyZ_of_right_side_cam)

            # Store ground truth
            finger_lmks_gt = np.reshape(finger_lmks_gt, (-1, 3))  # shape: (20, 3)
            landmarks_gt = np.vstack((wrist_gt[None, :], finger_lmks_gt))  # shape: (21, 3)
            lmks_gt_through_frames.append(landmarks_gt) 

            count += 1

        if count == 500:
            break

        time.sleep(0.001)

    print('----------------------- Stop storing ----------------------- ')   
    print('----------------------- Start saving ----------------------- ')   


    now = datetime.now()
    file_name = "hand_landmarks_{}_{}_{}_{}_{}.npz".format(now.year,
                                                           now.month,
                                                           now.day,
                                                           now.hour,
                                                           now.minute)
    des_path = os.path.join(DATA_FOLDER, file_name)

    raw_xyZ_of_opposite_cam_through_frames = np.array(xyZ_of_opposite_cam_through_frames)  # shape: (N, 21, 3)
    raw_xyZ_of_rightside_cam_through_frames = np.array(xyZ_of_rightside_cam_through_frames)  # shape: (N, 21, 3)
    landmarks_output_through_frames = np.array(lmks_gt_through_frames)  # shape: (N, 21, 3)

    # --------------- Save to .npz to run offline -------------------
    np.savez(des_path, 
             raw_xyZ_of_opposite_cam=raw_xyZ_of_opposite_cam_through_frames,
             raw_xyZ_of_rightside_cam=raw_xyZ_of_rightside_cam_through_frames,
             landmarks_output=landmarks_output_through_frames)

    # --------------- Save to .csv to train model -------------------
    raw_XYZ_of_opposite_cam_through_frames = xyZ_to_XYZ(raw_xyZ_of_opposite_cam_through_frames, rs_ins)  # shape: (N, 21, 3)
    raw_XYZ_of_right_side_cam_through_frames = xyZ_to_XYZ(raw_xyZ_of_rightside_cam_through_frames, oak_ins)  # shape: (N, 21, 3)
    raw_XYZ_of_right_side_cam_in_opposite_cam_through_frames = transform_to_another_coordinate(raw_XYZ_of_right_side_cam_through_frames,
                                                                                               oak_2_rs_mat)  # shape: (N, 21, 3)

    lmks_input = np.concatenate([raw_XYZ_of_opposite_cam_through_frames,
                                 raw_XYZ_of_right_side_cam_in_opposite_cam_through_frames], axis=1)  # shape: (N, 42, 3)
    lmks_input = lmks_input.reshape(lmks_input.shape[0], -1)  # shape: (N, 42 * 3)
    lmks_output = landmarks_output_through_frames.reshape(landmarks_output_through_frames.shape[0], -1)  # shape: (N, 21 * 3)

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(lmks_input, lmks_output, test_size=0.2, random_state=42)

    train_data = np.concatenate([X_train, Y_train], axis=1)
    test_data = np.concatenate([X_test, Y_test], axis=1)

    train_path_name = "train_hand_landmarks_{}_{}_{}_{}_{}.csv".format(now.year,
                                                                       now.month,
                                                                       now.day,
                                                                       now.hour,
                                                                       now.minute)
    train_path = os.path.join(DATA_FOLDER, train_path_name)
    write_data_to_csv(train_path, train_data)

    test_path_name = "test_hand_landmarks_{}_{}_{}_{}_{}.csv".format(now.year,
                                                                      now.month,
                                                                      now.day,
                                                                      now.hour,
                                                                      now.minute)
    test_path = os.path.join(DATA_FOLDER, test_path_name)
    write_data_to_csv(test_path, test_data)

    print('----------------------- Save done ----------------------- ')   