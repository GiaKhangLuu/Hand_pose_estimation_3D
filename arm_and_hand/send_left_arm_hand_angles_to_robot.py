import socket
import math
import numpy as np
import time

from angle_pid import AnglePID
from send_leftarm_data_to_robot import max_velocity_each_joint as max_velocity_of_arm_joints
from send_leftarm_data_to_robot import max_acce_each_joint as max_acce_of_arm_joints
from send_leftarm_data_to_robot import joints_pid_config as arm_joints_pid_config
from send_lefthand_data_to_robot import max_velocity_each_finger_joint as max_velocity_of_hand_joints
from send_lefthand_data_to_robot import max_acce_each_finger_joint as max_acce_of_hand_joints
from send_lefthand_data_to_robot import joints_pid_config as hand_joints_pid_config
from send_lefthand_data_to_robot import degree_to_radian

CLIENT_SOCKET = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#server_ip = "127.0.0.1"
SERVER_IP = "192.168.0.155"
SERVER_PORT = 12345
global_timestamp = 0

FPS = 100
TIME_SLEEP = 1 / FPS

NUM_ARM_ANGLES = 6
NUM_HAND_ANGLES = 15
NUM_ANGLES_EACH_FINGER = 3
TOTAL_ANGLES = NUM_ARM_ANGLES + NUM_HAND_ANGLES
FINGERS_NAME = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]

current_angles = np.array([0] * TOTAL_ANGLES, dtype=np.float64)

def send_angles_to_robot_using_pid(target_angles_queue=None, degree=True):
    """
    TODO: Doc.
    """

    global current_angles
    global global_timestamp

    pid_manager = []
    for angle_idx in range(TOTAL_ANGLES):
        if angle_idx < NUM_ARM_ANGLES:
            pid_conf = arm_joints_pid_config["joint{}".format(angle_idx + 1)]
            v_max = max_velocity_of_arm_joints[angle_idx]
            a_max = max_acce_of_arm_joints[angle_idx]
        else:
            temp = angle_idx - NUM_ARM_ANGLES
            finger_idx = temp // NUM_ANGLES_EACH_FINGER
            joint_idx = temp % NUM_ANGLES_EACH_FINGER
            pid_conf = hand_joints_pid_config[FINGERS_NAME[finger_idx]]["joint{}".format(joint_idx + 1)]
            hand_velo_acce_idx = joint_idx + (NUM_ANGLES_EACH_FINGER * finger_idx)
            v_max = max_velocity_of_hand_joints[hand_velo_acce_idx]
            a_max = max_acce_of_hand_joints[hand_velo_acce_idx]
        setpoint = current_angles[angle_idx]
        pid = AnglePID(Kp=pid_conf["Kp"], Ki=pid_conf["Ki"], Kd=pid_conf["Kd"],
            setpoint=setpoint, v_max=v_max, a_max=a_max, dt=TIME_SLEEP)
        pid_manager.append(pid)

    start_time = time.time()
    while True:
        if not target_angles_queue.empty():
            target_angles, timestamp = target_angles_queue.get()
            global_timestamp = timestamp
            for i in range(TOTAL_ANGLES):
                pid_manager[i].update_setpoint(target_angles[i])

        next_angles = np.zeros_like(current_angles, dtype=np.float64)
        for i in range(TOTAL_ANGLES):
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
        CLIENT_SOCKET.sendto(udp_mess.encode(), (SERVER_IP, SERVER_PORT))

        #print(udp_mess)

        time.sleep(TIME_SLEEP)