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
server_port = 12345

fps = 60
time_sleep = 1 / fps

current_angles = np.array([0, 0, 0, 0, 0, 0])
target_angles = np.array([0, 0, 0, 0, 0, 0])

write_angle_to_file = True

global_timestamp = 0

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
            #continue
        else:
            desired_end_angles, timestamp = target_angles_queue.get()
            global_timestamp = timestamp
            target_angles = desired_end_angles.copy()

        start_angles = current_angles.copy()
        next_angles = lerp(start_angles, desired_end_angles, lerp_factor)
        #next_angles = desired_end_angles
        current_angles = next_angles
        if degree:
            next_rad_angles = degree_to_radian(next_angles)
        next_rad_angles = next_rad_angles.tolist()

        end_time = time.time()
        delta_t = end_time - start_time
        start_time = end_time

        next_rad_angles.append(delta_t)
        udp_mess = str(next_rad_angles)
        client_socket.sendto(udp_mess.encode(), (server_ip, server_port))

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