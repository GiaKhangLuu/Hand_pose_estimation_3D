import time
import numpy as np
import sys
import os
from datetime import datetime

CURR_FOLDER = os.path.dirname(os.path.abspath(__file__))

now = datetime.now()
year, month, day = str(now.year), str(now.month), str(now.day)
month = "0{}".format(month) if len(month) == 1 else month
day = "0{}".format(day) if len(day) == 1 else day
DATA_FOLDER = os.path.join(CURR_FOLDER, "data", "{}_{}_{}".format(year, month, day))

if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

sys.path.append(os.path.join(CURR_FOLDER, '..'))

def write_lnmks_to_file(lmks_queue):
    count = 0

    fused_landmarks = []

    # -------------------- WARM UP -------------------- 
    time.sleep(20)

    while count >= 0:

        if not lmks_queue.empty():
            print('----------------------- Storing hand landmarks ----------------------- ')

            # landmarks: (N, 3)
            landmarks = lmks_queue.get() 

            fused_landmarks.append(landmarks) 

            count += 1

        if count == 500:
            break

        time.sleep(0.001)

    print('----------------------- Stop storing ----------------------- ')   
    print('----------------------- Start saving ----------------------- ')   


    now = datetime.now()
    file_name = "arm_landmarks_{}_{}_{}_{}_{}.npz".format(now.year,
                                                          now.month,
                                                          now.day,
                                                          now.hour,
                                                          now.minute)
    des_path = os.path.join(DATA_FOLDER, file_name)



    fused_landmarks = np.array(fused_landmarks)  # shape: (N, 21, 3)

    # --------------- Save to .npz to run offline -------------------
    np.savez(des_path, 
             fused_landmarks=fused_landmarks)

    # --------------- Save to .csv to train model -------------------

    # Save to .csv will be performed later

    print('----------------------- Save done ----------------------- ')   