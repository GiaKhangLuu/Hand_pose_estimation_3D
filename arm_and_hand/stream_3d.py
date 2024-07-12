import sys
import os

#CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
#sys.path.append(os.path.join(CURRENT_DIR, '..'))

import numpy as np
import open3d as o3d
import threading
import time

from angle_calculation import (calculate_angle_j1,
    calculate_angle_j2,
    calculate_angle_j3,
    calculate_angle_j4,
    calculate_rotation_matrix_to_compute_angle_of_j3_and_j4)

def visualize_arm(lmks_queue,
    landmark_dictionary, 
    show_left_arm_j1=False,
    show_left_arm_j2=False,
    show_left_arm_j3=False,
    show_left_arm_j4=False,
    show_left_arm_j5=False,
    draw_xyz=False,
    visualize_with_hand=False,
    joint_vector_color=None,
    ref_vector_color=None):
    x = np.array([[0, 0, 0],
                  [500, 0, 0],
                  [0, 500, 0],
                  [0, 0, 500]])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(x)

    lines = [[0, 1], [0, 2], [0, 3]]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(x),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(line_set)
    
    # Main loop
    while True:
        if not lmks_queue.empty():
            pts = lmks_queue.get()
            if visualize_with_hand:
                lines = [[0, 1], [1, 9], [0, 6], [0, 7], [6, 8], [7, 8],
                    [9, 10], [9, 14], [18, 22], [22, 26], [14, 18], [9, 26],
                    [10, 11], [11, 12], [12, 13],
                    [14, 15], [15, 16], [16, 17], 
                    [18, 19], [19, 20], [20, 21], 
                    [22, 23], [23, 24], [24, 25],
                    [26, 27], [27, 28], [28, 29]]
            else:
                lines = [[0, 1], [1, 2], [2, 3], [2, 4], [2, 5], [3, 4],
                    [0, 6], [0, 7],
                    [6, 8], [7, 8]]

            colors = [[0, 0, 0] for i in range(len(lines))]

            if show_left_arm_j3:  # Debugging calculating joint 3
                _, _, angle_j1 = calculate_angle_j1(pts, landmark_dictionary)
                _, _, angle_j2 = calculate_angle_j2(pts, landmark_dictionary)
                trans_mat, trans_mat_inv = calculate_rotation_matrix_to_compute_angle_of_j3_and_j4(pts, angle_j1, angle_j2, landmark_dictionary)
                b_prime_proj, b_prime, _ = calculate_angle_j3(pts, trans_mat_inv, landmark_dictionary)

                b_prime_in_world_to_plot = np.matmul(trans_mat, b_prime.T)
                b_prime_in_world_to_plot = b_prime_in_world_to_plot.T
                b_prime_proj_in_world_to_plot = np.matmul(trans_mat, b_prime_proj.T)
                b_prime_proj_in_world_to_plot = b_prime_proj_in_world_to_plot.T

                pts = np.concatenate([pts, [b_prime_in_world_to_plot, b_prime_proj_in_world_to_plot]], axis=0)
                last_index = pts.shape[0] - 1
                lines.extend([[0, last_index - 1], [0, last_index]])
                colors.extend([joint_vector_color, ref_vector_color])

            if show_left_arm_j4: # Debugging calculating joint 4
                _, _, angle_j1 = calculate_angle_j1(pts, landmark_dictionary)
                _, _, angle_j2 = calculate_angle_j2(pts, landmark_dictionary)
                trans_mat, trans_mat_inv = calculate_rotation_matrix_to_compute_angle_of_j3_and_j4(pts, angle_j1, angle_j2, landmark_dictionary)
                a, b, _ = calculate_angle_j4(pts, trans_mat, trans_mat_inv, landmark_dictionary)

                b_in_original_coor = np.matmul(trans_mat, b.T)
                b_in_original_coor = b_in_original_coor.T
                a_in_original_coor = np.matmul(trans_mat, a.T)
                a_in_original_coor = a_in_original_coor.T

                pts = np.concatenate([pts, [b_in_original_coor, a_in_original_coor * 20]], axis=0)
                last_index = pts.shape[0] - 1
                lines.extend([[0, last_index - 1], [0, last_index]])
                colors.extend([joint_vector_color, ref_vector_color])

            pcd.points = o3d.utility.Vector3dVector(pts)
            line_set.points = o3d.utility.Vector3dVector(pts)  # Update the points
            line_set.lines = o3d.utility.Vector2iVector(lines)  # Update the lines
            line_set.colors = o3d.utility.Vector3dVector(colors)

            # Update the visualization
            vis.update_geometry(pcd)
            vis.update_geometry(line_set)
        vis.poll_events()
        vis.update_renderer()        

        time.sleep(0.01)  # Set time sleep here is importance, the higher time.sleep parameter is (unit is second), the faster the main thread can process

    vis.destroy_window()

def visualize_hand(lmks_queue):
    x = np.array([[0, 0, 0],
                  [500, 0, 0],
                  [0, 500, 0],
                  [0, 0, 500]])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(x)

    lines = [[0, 1], [0, 2], [0, 3]]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(x),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(line_set)
    
    # Main loop
    while True:
        if not lmks_queue.empty():
            pts = lmks_queue.get()
            lines = [[0, 1], [1, 2], [2, 3], [3, 4],
                     [0, 5], [5, 6], [6, 7], [7, 8],
                     [0, 9], [9, 10], [10, 11], [11, 12],
                     [0, 13], [13, 14], [14, 15], [15, 16],
                     [0, 17], [17, 18], [18, 19], [19, 20]]
            colors = [[0, 0, 0] for i in range(len(lines))]

            pcd.points = o3d.utility.Vector3dVector(pts)
            line_set.points = o3d.utility.Vector3dVector(pts)  # Update the points
            line_set.lines = o3d.utility.Vector2iVector(lines)  # Update the lines
            line_set.colors = o3d.utility.Vector3dVector(colors)

            # Update the visualization
            vis.update_geometry(pcd)
            vis.update_geometry(line_set)
        vis.poll_events()
        vis.update_renderer()        

        time.sleep(0.1)

    vis.destroy_window()