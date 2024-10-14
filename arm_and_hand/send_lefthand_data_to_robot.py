import socket
import math
import numpy as np
import time

from angle_pid import AnglePID
from csv_writer import (create_csv, 
    append_to_csv, 
    fusion_csv_columns_name, 
    split_train_test_val,
    left_arm_angles_name)

client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#server_ip = "127.0.0.1"
#server_ip = "192.168.0.143"
server_ip = "192.168.0.126"
server_port = 12345
global_timestamp = 0

FPS = 60
TIME_SLEEP = 1 / FPS

num_fingers_angle = 15
num_joint_each_finger = 3
current_angles = np.array([0] * num_fingers_angle, dtype=np.float64)
target_angles = np.array([0] * num_fingers_angle, dtype=np.float64)

leftfinger_joint11_max_velo = 1.9  * (180 / math.pi)
leftfinger_joint12_max_velo = 1.256 * (180 / math.pi)
leftfinger_joint13_max_velo = 1.256 * (180 / math.pi)
leftfinger_joint21_max_velo = 3.0 * (180 / math.pi)
leftfinger_joint22_max_velo = 3.0 * (180 / math.pi)
leftfinger_joint23_max_velo = 3.0 * (180 / math.pi)
leftfinger_joint31_max_velo = 3.0 * (180 / math.pi)
leftfinger_joint32_max_velo = 3.0 * (180 / math.pi)
leftfinger_joint33_max_velo = 3.0 * (180 / math.pi)
leftfinger_joint41_max_velo = 3.0 * (180 / math.pi)
leftfinger_joint42_max_velo = 3.0 * (180 / math.pi)
leftfinger_joint43_max_velo = 3.0 * (180 / math.pi)
leftfinger_joint51_max_velo = 3.0 * (180 / math.pi)
leftfinger_joint52_max_velo = 3.0 * (180 / math.pi)
leftfinger_joint53_max_velo = 3.0 * (180 / math.pi)

leftfinger_joint11_max_acce = 8.0 * (180 / math.pi)
leftfinger_joint12_max_acce = 8.0 * (180 / math.pi)
leftfinger_joint13_max_acce = 8.0 * (180 / math.pi)
leftfinger_joint21_max_acce = 8.0 * (180 / math.pi)
leftfinger_joint22_max_acce = 8.0 * (180 / math.pi)
leftfinger_joint23_max_acce = 8.0 * (180 / math.pi)
leftfinger_joint31_max_acce = 8.0 * (180 / math.pi)
leftfinger_joint32_max_acce = 8.0 * (180 / math.pi)
leftfinger_joint33_max_acce = 8.0 * (180 / math.pi)
leftfinger_joint41_max_acce = 8.0 * (180 / math.pi)
leftfinger_joint42_max_acce = 8.0 * (180 / math.pi)
leftfinger_joint43_max_acce = 8.0 * (180 / math.pi)
leftfinger_joint51_max_acce = 8.0 * (180 / math.pi)
leftfinger_joint52_max_acce = 8.0 * (180 / math.pi)
leftfinger_joint53_max_acce = 8.0 * (180 / math.pi)

# Thumb finger
j11_Kp = 40
j11_Ki = 70
j11_Kd = 15

j12_Kp = 40
j12_Ki = 70
j12_Kd = 15

j13_Kp = 40
j13_Ki = 70
j13_Kd = 15

# Index finger
j21_Kp = 40
j21_Ki = 70
j21_Kd = 15

j22_Kp = 40
j22_Ki = 70
j22_Kd = 15

j23_Kp = 40
j23_Ki = 70
j23_Kd = 15

# Middle finger
j31_Kp = 40
j31_Ki = 70
j31_Kd = 15

j32_Kp = 40
j32_Ki = 70
j32_Kd = 15

j33_Kp = 40
j33_Ki = 70
j33_Kd = 15

# Ring finger
j41_Kp = 40
j41_Ki = 70
j41_Kd = 15

j42_Kp = 40
j42_Ki = 70
j42_Kd = 15

j43_Kp = 40
j43_Ki = 70
j43_Kd = 15

# Pinky finger
j51_Kp = 40
j51_Ki = 70
j51_Kd = 15

j52_Kp = 40
j52_Ki = 70
j52_Kd = 15

j53_Kp = 40
j53_Ki = 70
j53_Kd = 15

joints_pid_config = {
    "THUMB": {
        "joint1": {
            "Kp": j11_Kp,
            "Ki": j11_Ki,
            "Kd": j11_Kd
        },
        "joint2": {
            "Kp": j12_Kp,
            "Ki": j12_Ki,
            "Kd": j12_Kd
        },
        "joint3": {
            "Kp": j13_Kp,
            "Ki": j13_Ki,
            "Kd": j13_Kd
        }
    },
    "INDEX": {
        "joint1": {
            "Kp": j21_Kp,
            "Ki": j21_Ki,
            "Kd": j21_Kd
        },
        "joint2": {
            "Kp": j22_Kp,
            "Ki": j22_Ki,
            "Kd": j22_Kd
        },
        "joint3": {
            "Kp": j23_Kp,
            "Ki": j23_Ki,
            "Kd": j23_Kd
        }
    },
    "MIDDLE": {
        "joint1": {
            "Kp": j31_Kp,
            "Ki": j31_Ki,
            "Kd": j31_Kd
        },
        "joint2": {
            "Kp": j32_Kp,
            "Ki": j32_Ki,
            "Kd": j32_Kd
        },
        "joint3": {
            "Kp": j33_Kp,
            "Ki": j33_Ki,
            "Kd": j33_Kd
        }
    },
    "RING": {
        "joint1": {
            "Kp": j41_Kp,
            "Ki": j41_Ki,
            "Kd": j41_Kd
        },
        "joint2": {
            "Kp": j42_Kp,
            "Ki": j42_Ki,
            "Kd": j42_Kd
        },
        "joint3": {
            "Kp": j43_Kp,
            "Ki": j43_Ki,
            "Kd": j43_Kd
        }
    },
    "PINKY": {
        "joint1": {
            "Kp": j51_Kp,
            "Ki": j51_Ki,
            "Kd": j51_Kd
        },
        "joint2": {
            "Kp": j52_Kp,
            "Ki": j52_Ki,
            "Kd": j52_Kd
        },
        "joint3": {
            "Kp": j53_Kp,
            "Ki": j53_Ki,
            "Kd": j53_Kd
        }
    }
}

def degree_to_radian(degree):
    radian = degree * math.pi / 180 
    return radian

max_velocity_each_finger_joint = np.array([leftfinger_joint11_max_velo, 
    leftfinger_joint12_max_velo, 
    leftfinger_joint13_max_velo, 
    leftfinger_joint21_max_velo, 
    leftfinger_joint22_max_velo, 
    leftfinger_joint23_max_velo, 
    leftfinger_joint31_max_velo, 
    leftfinger_joint32_max_velo, 
    leftfinger_joint33_max_velo, 
    leftfinger_joint41_max_velo, 
    leftfinger_joint42_max_velo, 
    leftfinger_joint43_max_velo, 
    leftfinger_joint51_max_velo, 
    leftfinger_joint52_max_velo, 
    leftfinger_joint53_max_velo], dtype=np.float64)
max_acce_each_finger_joint = np.array([leftfinger_joint11_max_acce,
    leftfinger_joint12_max_acce,
    leftfinger_joint13_max_acce,
    leftfinger_joint21_max_acce,
    leftfinger_joint22_max_acce,
    leftfinger_joint23_max_acce,
    leftfinger_joint31_max_acce,
    leftfinger_joint32_max_acce,
    leftfinger_joint33_max_acce,
    leftfinger_joint41_max_acce,
    leftfinger_joint42_max_acce,
    leftfinger_joint43_max_acce,
    leftfinger_joint51_max_acce,
    leftfinger_joint52_max_acce,
    leftfinger_joint53_max_acce], dtype=np.float64)

def send_lefthand_fingers_udp_message_using_pid(target_angles_queue=None, degree=True):
    """
    Input:
        target_angles: numpy array, shape = (15,)
        degree: bool. Convert to radian if degree is True. Default = True
    """
    global client_socket
    global server_ip
    global server_port
    global TIME_SLEEP
    global current_angles
    global target_angles
    global global_timestamp
    global num_fingers_angle
    global num_joint_each_finger

    pid_manager = []
    for i, finger_name in enumerate(["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]):
        for j in range(num_joint_each_finger):
            joint_idx = j + (num_joint_each_finger * i)
            pid_conf = joints_pid_config[finger_name]["joint{}".format(j + 1)]
            setpoint = current_angles[joint_idx]
            v_max = max_velocity_each_finger_joint[joint_idx]
            a_max = max_acce_each_finger_joint[joint_idx]
            pid = AnglePID(Kp=pid_conf["Kp"], Ki=pid_conf["Ki"], Kd=pid_conf["Kd"],
                setpoint=setpoint, v_max=v_max, a_max=a_max, dt=TIME_SLEEP)
            pid_manager.append(pid)
    
    start_time = time.time()
    while True:
        if not target_angles_queue.empty():
            target_angles, timestamp = target_angles_queue.get()
            global_timestamp = timestamp
            for i in range(num_fingers_angle):
                pid_manager[i].update_setpoint(target_angles[i])

        next_angles = np.zeros_like(current_angles, dtype=np.float64)
        for i in range(num_fingers_angle):
            angle_pid = pid_manager[i]
            ji_current_angle = current_angles[i]
            ji_next_angle, _ = angle_pid.update(ji_current_angle)
            next_angles[i] = ji_next_angle

        current_angles = next_angles.copy()

        end_time = time.time()
        delta_t = end_time - start_time
        start_time = end_time

        if degree:
            next_rad_angles = degree_to_radian(next_angles)
        next_rad_angles = next_rad_angles.tolist()

        next_rad_angles.append(delta_t)
        udp_mess = str(next_rad_angles)
        client_socket.sendto(udp_mess.encode(), (server_ip, server_port))

        #print(udp_mess)

        time.sleep(TIME_SLEEP)

