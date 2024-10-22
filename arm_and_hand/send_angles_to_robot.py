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
SERVER_IP = "192.168.0.155"
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

current_angles = np.array([0] * TOTAL_ANGLES, dtype=np.float64)

def degree_to_radian(degree):
    radian = degree * math.pi / 180 
    return radian

def send_angles_to_robot_using_pid(target_angles_queue=None, degree=True):
    """
    TODO: Doc.
    As convention, our publish data should be: [*left_arm_angles, *right_arm_angles, *left_hand_angles, *right_hand_angles]
    """
    
    global current_angles 
    global global_timestamp
    
    pid_manager = [None] * TOTAL_ANGLES
    for angle_idx in range(TOTAL_ANGLES):
        if angle_idx < NUM_LEFT_ARM_ANGLES:
            pid_conf = left_arm_joints_pid_config["joint{}".format(angle_idx + 1)]
            v_max = left_arm_max_velocity_container[angle_idx]
            a_max = left_arm_max_acce_container[angle_idx]
        elif NUM_LEFT_ARM_ANGLES <= angle_idx < NUM_LEFT_ARM_ANGLES + NUM_RIGHT_ARM_ANGLES:
            right_arm_joint_i = angle_idx - NUM_LEFT_ARM_ANGLES
            pid_conf = right_arm_joints_pid_config["joint{}".format(right_arm_joint_i + 1)]
            v_max = right_arm_max_velocity_container[right_arm_joint_i]
            a_max = right_arm_max_acce_container[right_arm_joint_i]
        elif NUM_LEFT_ARM_ANGLES + NUM_RIGHT_ARM_ANGLES <= angle_idx < TOTAL_ANGLES:
            temp = angle_idx - (NUM_LEFT_ARM_ANGLES + NUM_RIGHT_ARM_ANGLES)
            finger_idx = temp // NUM_ANGLES_EACH_FINGER
            joint_idx = temp % NUM_ANGLES_EACH_FINGER
            pid_conf = left_hand_joints_pid_config[FINGERS_NAME[finger_idx]]["joint{}".format(joint_idx + 1)]
            left_hand_velo_acc_idx = joint_idx + (NUM_ANGLES_EACH_FINGER * finger_idx)         
            v_max = left_hand_max_velocity_container[left_hand_velo_acc_idx]
            a_max = left_hand_max_acce_container[left_hand_velo_acc_idx]
        else:
            return None
        setpoint = current_angles[angle_idx]
        pid = AnglePID(Kp=pid_conf["Kp"], Ki=pid_conf["Ki"], Kd=pid_conf["Kd"],
            setpoint=setpoint, v_max=v_max, a_max=a_max, dt=TIME_SLEEP)
        pid_manager[angle_idx] = pid     
    
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

        data = {
            "head": [0, 0],
            "left_arm": next_rad_angles[:NUM_LEFT_ARM_ANGLES],
            "right_arm": next_rad_angles[NUM_LEFT_ARM_ANGLES:NUM_LEFT_ARM_ANGLES + NUM_RIGHT_ARM_ANGLES],
            "left_fingers": next_rad_angles[NUM_LEFT_ARM_ANGLES + NUM_RIGHT_ARM_ANGLES:]
        }
        
        udp_mess = str(data)
        CLIENT_SOCKET.sendto(udp_mess.encode(), (SERVER_IP, SERVER_PORT))

        print(udp_mess)

        time.sleep(TIME_SLEEP)