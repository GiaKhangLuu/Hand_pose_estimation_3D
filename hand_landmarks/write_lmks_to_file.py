import numpy as np
import os
import time

curFolder = os.path.dirname(os.path.abspath(__file__))

def write_lnmks_to_file(lmks_queue):
    count = 0
    lmks_all_frame = []
    while count >= 0:
        count += 1
        if count >= 5000:
            print('----------------------- Saving ----------------------- ')

            if not lmks_queue.empty():
                wrist, finger_lmks = lmks_queue.get()
                finger_lmks = np.reshape(finger_lmks, (-1, 3))
                pts = np.vstack((wrist[None, :], finger_lmks))
                lmks_all_frame.append(pts)

        if count == 5000 * 2:
            count = -1

        time.sleep(0.001)

    print('----------------------- Stop saving ----------------------- ')   

    des_path = os.path.join(curFolder, 'hand_landmarks.npz')
    np.savez(des_path, landmarks=np.array(lmks_all_frame))

    print('----------------------- Save done ----------------------- ')   