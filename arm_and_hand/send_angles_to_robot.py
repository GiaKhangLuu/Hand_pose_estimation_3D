import socket
import math
import numpy as np
import time

from angle_pid import AnglePID
from configuration.joints_limit import (
    head_max_velocity_container, head_max_acce_container,
    left_arm_max_velocity_container, left_arm_max_acce_container,
    right_arm_max_velocity_container, right_arm_max_acce_container,
    left_hand_max_velocity_container, left_hand_max_acce_container
)
from configuration.pid_config import (
    head_joints_pid_config,
    left_arm_joints_pid_config,
    right_arm_joints_pid_config, 
    left_hand_joints_pid_config
)

CLIENT_SOCKET = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
SERVER_IP = "192.168.0.190"
SERVER_PORT = 12345
global_timestamp = 0

FPS = 100
TIME_SLEEP = 1 / FPS

NUM_HEAD_ANGLES = 2
NUM_LEFT_ARM_ANGLES = 6
NUM_RIGHT_ARM_ANGLES = 6
NUM_LEFT_HAND_ANGLES = 15
NUM_ANGLES_EACH_FINGER = 3
#TOTAL_ANGLES = NUM_HEAD_ANGLES + NUM_LEFT_ARM_ANGLES + NUM_RIGHT_ARM_ANGLES + NUM_LEFT_HAND_ANGLES
TOTAL_ANGLES = NUM_LEFT_ARM_ANGLES + NUM_RIGHT_ARM_ANGLES + NUM_LEFT_HAND_ANGLES
FINGERS_NAME = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]

VELOCITY_MANAGER = {
    "head": head_max_velocity_container,
    "left_arm": left_arm_max_velocity_container,
    "right_arm": right_arm_max_velocity_container,
    "left_fingers": left_hand_max_velocity_container
}

ACCELERATION_MANAGER = {
    "head": head_max_acce_container,
    "left_arm": left_arm_max_acce_container,
    "right_arm": right_arm_max_acce_container,
    "left_fingers": left_hand_max_acce_container
}

PID_CONFIG_MANAGER = {
    "head": head_joints_pid_config,
    "left_arm": left_arm_joints_pid_config,
    "right_arm": right_arm_joints_pid_config,
    "left_fingers": left_hand_joints_pid_config
}

pid_manager = {
    "head": [None] * NUM_HEAD_ANGLES,
    "left_arm": [None] * NUM_LEFT_ARM_ANGLES,
    "right_arm": [None] * NUM_RIGHT_ARM_ANGLES,
    "left_fingers": [None] * NUM_LEFT_HAND_ANGLES
}

current_angles_for_parts = {
    "head": np.zeros(NUM_HEAD_ANGLES, dtype=np.float64),
    "left_arm": np.zeros(NUM_LEFT_ARM_ANGLES, dtype=np.float64),
    "right_arm": np.zeros(NUM_RIGHT_ARM_ANGLES, dtype=np.float64),
    "left_fingers": np.zeros(NUM_LEFT_HAND_ANGLES, dtype=np.float64)
}

def degree_to_radian(degree):
    radian = degree * math.pi / 180 
    return radian

def send_angles_to_robot_using_pid(target_angles_queue=None, degree=True):
    """
    TODO: Doc.
    As convention, our publish data should be: [*left_arm_angles, *right_arm_angles, *left_hand_angles, *right_hand_angles]
    """
    
    global current_angles_for_parts 
    global global_timestamp

    for part in pid_manager.keys():
        for joint_i in range(len(pid_manager[part])):
            v_max = VELOCITY_MANAGER[part][joint_i]
            a_max = ACCELERATION_MANAGER[part][joint_i]
            setpoint = current_angles_for_parts[part][joint_i]
            if part != "left_fingers":
                pid_conf = PID_CONFIG_MANAGER[part]["joint{}".format(joint_i + 1)]
            else:
                finger_i = joint_i // NUM_ANGLES_EACH_FINGER
                joint_of_finger_i = joint_i % NUM_ANGLES_EACH_FINGER
                pid_conf = PID_CONFIG_MANAGER[part][FINGERS_NAME[finger_i]]["joint{}".format(joint_of_finger_i + 1)]
            pid_manager[part][joint_i] = AnglePID(
                Kp=pid_conf["Kp"],
                Ki=pid_conf["Ki"],
                Kd=pid_conf["Kd"],
                setpoint=setpoint,
                v_max=v_max,
                a_max=a_max, 
                dt=TIME_SLEEP
            )

    start_time = time.time() 
    while True:
        if not target_angles_queue.empty():
            target_angles, timestamp = target_angles_queue.get()
            global_timestamp = timestamp

            for part in pid_manager.keys():
                for joint_i in range(len(pid_manager[part])):
                    target_angle = target_angles[part][joint_i]
                    pid_manager[part][joint_i].update_setpoint(target_angle)

        next_angles_for_parts = dict()
        for part in pid_manager.keys():
            next_angles_for_parts[part] = np.zeros(len(pid_manager[part]), dtype=np.float64)
            for joint_i in range(len(pid_manager[part])):
                angle_pid = pid_manager[part][joint_i]
                ji_curr_angle = current_angles_for_parts[part][joint_i]
                ji_next_angle, _ = angle_pid.update(ji_curr_angle)
                next_angles_for_parts[part][joint_i] = ji_next_angle
            
        current_angles_for_parts = next_angles_for_parts.copy()

        end_time = time.time()
        delta_t = end_time - start_time
        start_time = end_time

        current_rad_angles_for_parts = current_angles_for_parts.copy() 
        if degree:
            for part in current_angles_for_parts.keys():
                current_rad_angles_for_parts[part] = degree_to_radian(current_angles_for_parts[part].copy())
                current_rad_angles_for_parts[part] = current_rad_angles_for_parts[part].tolist()

        udp_mess = str(current_rad_angles_for_parts)
        CLIENT_SOCKET.sendto(udp_mess.encode(), (SERVER_IP, SERVER_PORT))

        print(udp_mess)

        time.sleep(TIME_SLEEP)