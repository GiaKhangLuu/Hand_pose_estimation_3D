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

FPS = 60
TIME_SLEEP = 1 / FPS

# We expect to use the same lerp_factor for delta_t's variances
STD_LERP_BY_FPS = 0.4 / (1 / FPS)

num_joint = 6
current_angles = np.array([0] * num_joint, dtype=np.float64)
target_angles = np.array([0] * num_joint, dtype=np.float64)

write_angle_to_file = False

global_timestamp = 0

joint1_max_velo = 1.5184 * (180 / math.pi)
joint2_max_velo = 1.5184 * (180 / math.pi)
joint3_max_velo = 2.1074 * (180 / math.pi)
joint4_max_velo = 2.1074 * (180 / math.pi)
joint5_max_velo = 3.3719 * (180 / math.pi)
joint6_max_velo = 3.3719 * (180 / math.pi)

joint1_max_acce = 3 * (180 / math.pi)
joint2_max_acce = 3 * (180 / math.pi)
joint3_max_acce = 4 * (180 / math.pi)
joint4_max_acce = 4 * (180 / math.pi)
joint5_max_acce = 6 * (180 / math.pi)
joint6_max_acce = 6 * (180 / math.pi)

j1_Kp = 30
j1_Ki = 70
j1_Kd = 10

j2_Kp = 30
j2_Ki = 70
j2_Kd = 10

j3_Kp = 40
j3_Ki = 70
j3_Kd = 15

j4_Kp = 40
j4_Ki = 70
j4_Kd = 15

j5_Kp = 40
j5_Ki = 70
j5_Kd = 15

j6_Kp = 40
j6_Ki = 70
j6_Kd = 15

joints_pid_config = {
    "joint1": {
        "Kp": j1_Kp,
        "Ki": j1_Ki,
        "Kd": j1_Kd
    },
    "joint2": {
        "Kp": j2_Kp,
        "Ki": j2_Ki,
        "Kd": j2_Kd
    },
    "joint3": {
        "Kp": j3_Kp,
        "Ki": j3_Ki,
        "Kd": j3_Kd
    },
    "joint4": {
        "Kp": j4_Kp,
        "Ki": j4_Ki,
        "Kd": j4_Kd
    },
    "joint5": {
        "Kp": j5_Kp,
        "Ki": j5_Ki,
        "Kd": j5_Kd
    },
    "joint6": {
        "Kp": j6_Kp,
        "Ki": j6_Ki,
        "Kd": j6_Kd
    }
}

max_velocity_each_joint = np.array([joint1_max_velo, joint2_max_velo, joint3_max_velo, 
    joint4_max_velo, joint5_max_velo, joint6_max_velo], dtype=np.float64)
max_acce_each_joint = np.array([joint1_max_acce, joint2_max_acce, joint3_max_acce,
    joint4_max_acce, joint5_max_acce, joint6_max_acce], dtype=np.float64)

def lerp(start, end, delta_t, lerp_factor=0.5):
    """
    Linearly interpolates between start and end by t.
    Input:
        start: The starting value.
        end: The ending value.
        t: A float between 0 and 1 representing the interpolation factor.
    Output:
        Interpolated value.
    """
    lerp_rate = lerp_factor * delta_t
    return (1 - lerp_rate) * start + lerp_rate * end

def degree_to_radian(degree):
    radian = degree * math.pi / 180 
    return radian

def limit_position_by_velocity(start_angles, next_angles, delta_t):
    """
    TODO: Doc
    Input:
        start_angles (np.array): angle at the current position, shape = (6,), for 6 angles
        next_angles (np.array): angle at the next expected position, shape = (6,), for 6 angles
        delta_t (float): delta time
    Output:
        bound_next_angle (np.array): angle at the next position after bounding
    """
    global max_velocity_each_joint

    bound_next_angles = next_angles.copy()
    delta_s = next_angles - start_angles
    velocities = delta_s / delta_t

    assert velocities.shape[0] == max_velocity_each_joint.shape[0]

    mask = np.abs(velocities) > max_velocity_each_joint
    bound_angles = max_velocity_each_joint[mask] * np.sign(velocities)[mask] * delta_t + start_angles[mask]
    bound_next_angles[mask] = bound_angles
    return bound_next_angles

def send_leftarm_angles_udp_message_using_lerp(target_angles_queue=None, degree=True):
    """
    Input:
        target_angles: numpy array, shape = (6,)
        degree: bool. Convert to radian if degree is True. Default = True
    """
    global client_socket
    global server_ip
    global server_port
    global TIME_SLEEP
    global current_angles
    global target_angles
    global global_timestamp

    if write_angle_to_file:
        create_csv("/home/giakhang/Desktop/debug_lerp.csv", left_arm_angles_name)

    global STD_LERP_BY_FPS
    start_time = time.time()
    while True:
        if target_angles_queue.empty():
            desired_end_angles = target_angles.copy()
        else:
            desired_end_angles, timestamp = target_angles_queue.get()
            global_timestamp = timestamp
            target_angles = desired_end_angles.copy()

        end_time = time.time()
        delta_t = end_time - start_time
        start_time = end_time

        start_angles = current_angles.copy().astype(np.float64)
        next_angles = lerp(start_angles, desired_end_angles, delta_t, STD_LERP_BY_FPS)
        next_angles = limit_position_by_velocity(start_angles=start_angles, 
            next_angles=next_angles, delta_t=delta_t)

        current_angles = next_angles.copy()
        if degree:
            next_rad_angles = degree_to_radian(next_angles)
        next_rad_angles = next_rad_angles.tolist()

        next_rad_angles.append(delta_t)
        udp_mess = str(next_rad_angles)
        client_socket.sendto(udp_mess.encode(), (server_ip, server_port))

        print(udp_mess)

        if write_angle_to_file:
            delta_t_ms = delta_t * 1000
            delta_fps = 1000 / delta_t_ms
            append_to_csv("/home/giakhang/Desktop/debug_lerp.csv",
                [global_timestamp, 
                round(delta_fps, 1), 
                round(current_angles[0], 3),
                round(current_angles[1], 3),
                round(current_angles[2], 3),
                round(current_angles[3], 3),
                round(current_angles[4], 3),
                round(current_angles[5], 3)])
            
        time.sleep(TIME_SLEEP) 

def send_leftarm_angles_udp_message_using_pid(target_angles_queue=None, degree=True):
    """
    Input:
        target_angles: numpy array, shape = (6,)
        degree: bool. Convert to radian if degree is True. Default = True
    """
    global client_socket
    global server_ip
    global server_port
    global TIME_SLEEP
    global current_angles
    global global_timestamp
    global num_joint

    if write_angle_to_file:
        create_csv("/home/giakhang/Desktop/debug_lerp.csv", left_arm_angles_name)

    pid_manager = []
    for i in range(num_joint):
        pid_conf = joints_pid_config["joint{}".format(i + 1)]
        setpoint = current_angles[i]
        v_max = max_velocity_each_joint[i]
        a_max = max_acce_each_joint[i]
        pid = AnglePID(Kp=pid_conf["Kp"], Ki=pid_conf["Ki"], Kd=pid_conf["Kd"],
            setpoint=setpoint, v_max=v_max, a_max=a_max, dt=TIME_SLEEP)
        pid_manager.append(pid)

    start_time = time.time()
    while True:
        if not target_angles_queue.empty():
            target_angles, timestamp = target_angles_queue.get()
            global_timestamp = timestamp
            for i in range(num_joint):
                pid_manager[i].update_setpoint(target_angles[i])

        next_angles = np.zeros_like(current_angles, dtype=np.float64)
        for i in range(num_joint):
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

        if write_angle_to_file and np.sum(np.abs(current_angles)) > 0:
            delta_t_ms = delta_t * 1000
            delta_fps = 1000 / delta_t_ms
            append_to_csv("/home/giakhang/Desktop/debug_lerp.csv",
                [global_timestamp, 
                round(delta_fps, 1), 
                round(pid_manager[0]._setpoint, 3),
                round(pid_manager[1]._setpoint, 3),
                round(pid_manager[2]._setpoint, 3),
                round(pid_manager[3]._setpoint, 3),
                round(pid_manager[4]._setpoint, 3),
                round(pid_manager[5]._setpoint, 3),
                round(current_angles[0], 3),
                round(current_angles[1], 3),
                round(current_angles[2], 3),
                round(current_angles[3], 3),
                round(current_angles[4], 3),
                round(current_angles[5], 3)])
            
        time.sleep(TIME_SLEEP) 