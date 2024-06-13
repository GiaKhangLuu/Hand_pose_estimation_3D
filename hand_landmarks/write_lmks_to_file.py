import numpy as np
import os
import time

curFolder = os.path.dirname(os.path.abspath(__file__))

def write_lnmks_to_file(lmks_queue, file_name='hand_landmarks_2024_06_12.npz'):
    count = 0
    lmks_gt_all_frame = []
    lmks_input_all_frame = []

    while count >= 0:
        # -------------------- WARM UP -------------------- 
        if count >= 5000:
            print('----------------------- Saving ----------------------- ')

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

        if count == 5000 * 2:
            count = -1

        time.sleep(0.001)

    print('----------------------- Stop saving ----------------------- ')   

    des_path = os.path.join(curFolder, file_name)
    np.savez(des_path, 
             landmarks_output=np.array(lmks_gt_all_frame), 
             landmarks_input=np.array(lmks_input_all_frame))

    print('----------------------- Save done ----------------------- ')   