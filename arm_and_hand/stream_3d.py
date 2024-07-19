import sys
import os

#CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
#sys.path.append(os.path.join(CURRENT_DIR, '..'))

import numpy as np
import open3d as o3d
import threading
import time
from scipy.spatial.transform import Rotation as R

from angle_calculation import (calculate_angle_j1,
    calculate_angle_j2,
    calculate_angle_j3,
    calculate_angle_j4,
    calculate_rotation_matrix_to_compute_angle_of_j3_and_j4,
    calculate_elbow_coords,
    calculate_wrist_coords,
    calculate_wrist_coords_in_elbow,
    calculate_angle_j5,
    calculate_angle_j6)

def visualize_arm(lmks_queue,
    landmark_dictionary, 
    show_left_arm_j1=False,
    show_left_arm_j2=False,
    show_left_arm_j3=False,
    show_left_arm_j4=False,
    show_left_arm_j5=False,
    show_left_arm_j6=False,
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
                shoulder_rot_mat, shoulder_rot_mat_inv = calculate_rotation_matrix_to_compute_angle_of_j3_and_j4(pts, angle_j1, angle_j2, landmark_dictionary)
                a, b, angle_j3 = calculate_angle_j3(pts, shoulder_rot_mat, shoulder_rot_mat_inv, landmark_dictionary)

                b_in_original_coor = np.matmul(shoulder_rot_mat, b.T)
                b_in_original_coor = b_in_original_coor.T
                a_in_original_coor = np.matmul(shoulder_rot_mat, a.T)
                a_in_original_coor = a_in_original_coor.T

                pts = np.concatenate([pts, [b_in_original_coor, a_in_original_coor * 20]], axis=0)
                last_index = pts.shape[0] - 1
                lines.extend([[0, last_index - 1], [0, last_index]])
                colors.extend([joint_vector_color, ref_vector_color])

            if show_left_arm_j4: # Debugging calculating joint 4
                _, _, angle_j1 = calculate_angle_j1(pts, landmark_dictionary)
                _, _, angle_j2 = calculate_angle_j2(pts, landmark_dictionary)
                shoulder_rot_mat, shoulder_rot_mat_inv = calculate_rotation_matrix_to_compute_angle_of_j3_and_j4(pts, angle_j1, angle_j2, landmark_dictionary)
                a, b, _ = calculate_angle_j4(pts, shoulder_rot_mat, shoulder_rot_mat_inv, landmark_dictionary)

                b_in_original_coor = np.matmul(shoulder_rot_mat, b.T)
                b_in_original_coor = b_in_original_coor.T
                a_in_original_coor = np.matmul(shoulder_rot_mat, a.T)
                a_in_original_coor = a_in_original_coor.T

                pts = np.concatenate([pts, [b_in_original_coor, a_in_original_coor * 20]], axis=0)
                last_index = pts.shape[0] - 1
                lines.extend([[0, last_index - 1], [0, last_index]])
                colors.extend([joint_vector_color, ref_vector_color])

            if show_left_arm_j5:  # Debugging calculating joint 5
                # Joint 1 and Joint 2
                _, _, angle_j1 = calculate_angle_j1(pts, landmark_dictionary)
                _, _, angle_j2 = calculate_angle_j2(pts, landmark_dictionary)

                # Joint 3 and Joint 4
                shoulder_rot_mat_in_w, shoulder_rot_mat_in_w_inv = calculate_rotation_matrix_to_compute_angle_of_j3_and_j4(pts, angle_j1, angle_j2, landmark_dictionary)
                _, _, angle_j3 = calculate_angle_j3(pts, shoulder_rot_mat_in_w, shoulder_rot_mat_in_w_inv, landmark_dictionary)
                _, _, angle_j4 = calculate_angle_j4(pts, shoulder_rot_mat_in_w, shoulder_rot_mat_in_w_inv, landmark_dictionary)

                # Joint 5
                elbow_coords_in_world, _ = calculate_elbow_coords(pts, landmark_dictionary, 
                    shoulder_rot_mat_in_w, angle_j3, angle_j4)
                wrist_coords_in_world = calculate_wrist_coords(pts, landmark_dictionary)
                wrist_coords_in_elbow = calculate_wrist_coords_in_elbow(wrist_coords_in_world, elbow_coords_in_world)
                rot_mat = R.from_matrix(wrist_coords_in_elbow)
                angle_j5 = calculate_angle_j5(rot_mat)

                print("angle_j5: ", angle_j5)

                pts = np.concatenate([pts, 
                    [
                     elbow_coords_in_world[:, 2] * 40,
                     wrist_coords_in_world[:, 2] * 40,
                     # Plot elbow and wrist coordinates at elbow and wrist
                     elbow_coords_in_world[:, 0] * 40 + pts[1, :],  
                     elbow_coords_in_world[:, 1] * 40 + pts[1, :], 
                     elbow_coords_in_world[:, 2] * 40 + pts[1, :],
                     wrist_coords_in_world[:, 0] * 40 + pts[9, :], 
                     wrist_coords_in_world[:, 1] * 40 + pts[9, :], 
                     wrist_coords_in_world[:, 2] * 40 + pts[9, :]]],
                    axis=0)
                last_index = pts.shape[0] - 1
                lines.extend([
                    [0, last_index - 7], [0, last_index - 6],
                    [1, last_index - 5], [1, last_index - 4], [1, last_index - 3],
                    [9, last_index - 2], [9, last_index - 1], [9, last_index]]),
                colors.extend([
                    ref_vector_color, joint_vector_color,
                    [1, 0, 0], [0, 1, 0], [0, 0, 1],
                    [1, 0, 0], [0, 1, 0], [0, 0, 1]])

            if show_left_arm_j6:  # Debugging calculating joint 6
                # Joint 1 and Joint 2
                _, _, angle_j1 = calculate_angle_j1(pts, landmark_dictionary)
                _, _, angle_j2 = calculate_angle_j2(pts, landmark_dictionary)

                # Joint 3 and Joint 4
                shoulder_rot_mat_in_w, shoulder_rot_mat_in_w_inv = calculate_rotation_matrix_to_compute_angle_of_j3_and_j4(pts, angle_j1, angle_j2, landmark_dictionary)
                _, _, angle_j3 = calculate_angle_j3(pts, shoulder_rot_mat_in_w, shoulder_rot_mat_in_w_inv, landmark_dictionary)
                _, _, angle_j4 = calculate_angle_j4(pts, shoulder_rot_mat_in_w, shoulder_rot_mat_in_w_inv, landmark_dictionary)

                # Joint 6
                elbow_coords_in_world, _ = calculate_elbow_coords(pts, landmark_dictionary, 
                    shoulder_rot_mat_in_w, angle_j3, angle_j4)
                wrist_coords_in_world = calculate_wrist_coords(pts, landmark_dictionary)
                wrist_coords_in_elbow = calculate_wrist_coords_in_elbow(wrist_coords_in_world, elbow_coords_in_world)
                rot_mat = R.from_matrix(wrist_coords_in_elbow)
                angle_j6 = calculate_angle_j6(rot_mat)

                print("angle_j6: ", angle_j6)

                pts = np.concatenate([pts, 
                    [
                     elbow_coords_in_world[:, 0] * 40,
                     wrist_coords_in_world[:, 0] * 40,
                     # Plot elbow and wrist coordinates at elbow and wrist
                     elbow_coords_in_world[:, 0] * 40 + pts[1, :],  
                     elbow_coords_in_world[:, 1] * 40 + pts[1, :], 
                     elbow_coords_in_world[:, 2] * 40 + pts[1, :],
                     wrist_coords_in_world[:, 0] * 40 + pts[9, :], 
                     wrist_coords_in_world[:, 1] * 40 + pts[9, :], 
                     wrist_coords_in_world[:, 2] * 40 + pts[9, :]]],
                    axis=0)
                last_index = pts.shape[0] - 1
                lines.extend([
                    [0, last_index - 7], [0, last_index - 6],
                    [1, last_index - 5], [1, last_index - 4], [1, last_index - 3],
                    [9, last_index - 2], [9, last_index - 1], [9, last_index]]),
                colors.extend([
                    ref_vector_color, joint_vector_color,
                    [1, 0, 0], [0, 1, 0], [0, 0, 1],
                    [1, 0, 0], [0, 1, 0], [0, 0, 1]])

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