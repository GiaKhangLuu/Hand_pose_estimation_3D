import numpy as np
import os
import time
from inputs import get_key

curFolder = os.path.dirname(os.path.abspath(__file__))
exit_flag = False

def check_for_exit():
    events = get_key()
    for event in events:
        if event.ev_type == 'Key' and event.ev_key == 'KEY_Q' and event.ev_state == 1:
            return True
    return False

def write_lnmks_to_file(lmks_queue, file_name='hand_landmarks_2024_06_12.npz'):
    count = 0
    lmks_gt_all_frame = []
    lmks_input_all_frame = []

    # -------------------- WARM UP -------------------- 
    time.sleep(20)

    while count >= 0:
        print('----------------------- Storing hand landmarks ----------------------- ')

        if not lmks_queue.empty():
            # raw_XYZ_of_opposite_cam: (21, 3)
            # raw_XYZ_of_right_side_cam_in_opposite_cam: (21, 3)
            # wrist_gt: (3,)
            # finger_lmks_gt: (5, 4, 3)
            raw_XYZ_of_opposite_cam, raw_XYZ_of_right_side_cam_in_opposite_cam, wrist_gt, finger_lmks_gt = lmks_queue.get() 

            # Save GTs
            finger_lmks_gt = np.reshape(finger_lmks_gt, (-1, 3))  # shape: (20, 3)
            landmarks_gt = np.vstack((wrist_gt[None, :], finger_lmks_gt))  # shape: (21, 3)
            lmks_gt_all_frame.append(landmarks_gt) 

            # Save input
            lmks_input = np.concatenate([raw_XYZ_of_opposite_cam, raw_XYZ_of_right_side_cam_in_opposite_cam], axis=0)  # shape: (42, 3)
            lmks_input_all_frame.append(lmks_input)

            count += 1

        if count == 500:
            break

        time.sleep(0.001)

    print('----------------------- Stop storing ----------------------- ')   
    print('----------------------- Start saving ----------------------- ')   

    des_path = os.path.join(curFolder, file_name)
    np.savez(des_path, 
             landmarks_output=np.array(lmks_gt_all_frame), 
             landmarks_input=np.array(lmks_input_all_frame))

    print('----------------------- Save done ----------------------- ')   