import socket
import math
import numpy as np
import time

from csv_writer import (create_csv, 
    append_to_csv, 
    fusion_csv_columns_name, 
    split_train_test_val,
    arm_angles_name)

client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_ip = "127.0.0.1"
#server_ip = "192.168.0.143"
server_port = 12345

fps = 10
time_sleep = 1 / fps

current_angles = np.array([0, 0, 0, 0, 0, 0])
target_angles = np.array([0, 0, 0, 0, 0, 0])

write_angle_to_file = False

global_timestamp = 0

joint1_max_velo = 1.5184 * (180 / math.pi)
joint2_max_velo = 1.5184 * (180 / math.pi)
joint3_max_velo = 2.1074 * (180 / math.pi)
joint4_max_velo = 2.1074 * (180 / math.pi)
joint5_max_velo = 3.3719 * (180 / math.pi)
joint6_max_velo = 3.3719 * (180 / math.pi)

max_velocity_each_joint = np.array([joint1_max_velo, joint2_max_velo, joint3_max_velo, 
    joint4_max_velo, joint5_max_velo, joint6_max_velo], dtype=np.float64)

def lerp(start, end, lerp_factor=0.5):
    """
    Linearly interpolates between start and end by t.
    Input:
        start: The starting value.
        end: The ending value.
        t: A float between 0 and 1 representing the interpolation factor.
    Output:
        Interpolated value.
    """
    return (1 - lerp_factor) * start + lerp_factor * end

def degree_to_radian(degree):
    radian = degree * math.pi / 180 
    return radian

def send_udp_message(target_angles_queue=None, degree=True, lerp_factor=0.5):
    """
    Input:
        target_angles: numpy array, shape = (6,)
        degree: bool. Convert to radian if degree is True. Default = True
    """
    global client_socket
    global server_ip
    global server_port
    global time_sleep
    global current_angles
    global target_angles
    global global_timestamp

    if write_angle_to_file:
        create_csv("/home/giakhang/Desktop/debug_lerp.csv", arm_angles_name)

    start_time = time.time()
    while True:
        if target_angles_queue.empty():
            desired_end_angles = target_angles.copy()
        else:
            desired_end_angles, timestamp = target_angles_queue.get()
            global_timestamp = timestamp
            target_angles = desired_end_angles.copy()

        start_angles = current_angles.copy().astype(np.float64)
        next_angles = lerp(start_angles, desired_end_angles, lerp_factor)

        end_time = time.time()
        delta_t = end_time - start_time
        start_time = end_time

        delta_s = next_angles - start_angles
        velocities = delta_s / delta_t

        assert velocities.shape[0] == max_velocity_each_joint.shape[0]

        mask = np.abs(velocities) > max_velocity_each_joint
        bound_angles = max_velocity_each_joint[mask] * np.sign(velocities)[mask] * delta_t + start_angles[mask]
        next_angles[mask] = bound_angles

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
            
        time.sleep(time_sleep) 