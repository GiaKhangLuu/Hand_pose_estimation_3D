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
    calculate_angle_j5,
    calculate_angle_j6,
    calculate_rotation_matrix_to_compute_angle_of_j1_and_j2,
    calculate_elbow_coordinate_wrt_origin,
    calculate_elbow_coordinate_wrt_shoulder,
    calculate_wrist_coordinate_wrt_origin,
    calculate_wrist_coordinate_wrt_elbow)

def visualize_arm(lmks_queue,
    landmark_dictionary, 
    show_left_arm_j1=False,
    show_left_arm_j2=False,
    show_left_arm_j3=False,
    show_left_arm_j4=False,
    show_left_arm_j5=False,
    show_left_arm_j6=False,
    draw_original_xyz=True,
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

    bounding_box = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array([[500,0 ,0], [0, 0, 0]])),
        lines=o3d.utility.Vector2iVector([[0, 0]])
    )
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(line_set)
    vis.add_geometry(bounding_box)
    
    # Main loop
    while True:
        if not lmks_queue.empty():
            pts, original_xyz = lmks_queue.get()  # pts.shape = (M, N), M = #vectors = 21, N = #features = 3. original_xyz.shape = (N, O), N = #features = 3, O = #vectors = 3 (xyz)
            if visualize_with_hand:
                lines = [[0, 1], [1, 5], [0, 2], [0, 3], [2, 4], 
                    [3, 4], [5, 6], [5, 10], [14, 18], [18, 22], 
                    [10, 14], [5, 22], [6, 7], [7, 8], [8, 9], 
                    [10, 11], [11, 12], [12, 13], [14, 15], [15, 16], 
                    [16, 17], [18, 19], [19, 20], [20, 21], [22, 23], 
                    [23, 24], [24, 25]]
            else:
                # For now, we dont get wrist, pinky, index and thumb landmark from POSE LANDMARKS DETECTION => We are not
                # able to draw without HAND DETECTION
                lines = [[0, 1], [1, 2], [2, 3], [2, 4], [2, 5], [3, 4],
                    [0, 6], [0, 7],
                    [6, 8], [7, 8]]

            colors = [[0, 0, 0] for i in range(len(lines))]

            if draw_original_xyz:
                assert original_xyz is not None
                x_unit, y_unit, z_unit = original_xyz[:, 0], original_xyz[:, 1], original_xyz[:, 2]
                pts = np.concatenate([pts, [x_unit * 40, y_unit * 40, z_unit * 40]], axis=0)
                last_index = pts.shape[0] - 1
                lines.extend([[0, last_index - 2], [0, last_index - 1], [0, last_index]])
                colors.extend([(1, 0, 0), (0, 1, 0), (0, 0, 1)])

            shoulder_coords_in_world = calculate_rotation_matrix_to_compute_angle_of_j1_and_j2(pts, landmark_dictionary, original_xyz)
            elbow_coordinate_wrt_origin = calculate_elbow_coordinate_wrt_origin(pts,
                landmark_dictionary, shoulder_coords_in_world)  # (3, O), O = number of vectors (xyz)
            wrist_coordinate_wrt_origin = calculate_wrist_coordinate_wrt_origin(pts,
                landmark_dictionary, elbow_coordinate_wrt_origin)  # (3, O), O = number of vectors (xyz)

            if show_left_arm_j1 or show_left_arm_j2:
                x_shoulder, y_shoulder, z_shoulder = shoulder_coords_in_world[:, 0],  shoulder_coords_in_world[:, 1], shoulder_coords_in_world[:, 2]
                shoulder_coords_in_world_rot_mat = R.from_matrix(shoulder_coords_in_world)

                pts = np.concatenate([pts, [x_shoulder * 40, y_shoulder * 40, z_shoulder * 40]], axis=0)
                last_index = pts.shape[0] - 1
                lines.extend([[0, last_index - 2], [0, last_index - 1], [0, last_index]])
                colors.extend([(1, 0, 0), (0, 1, 0), (0, 0, 1)])

                if show_left_arm_j1:
                    angle_j1 = calculate_angle_j1(shoulder_coords_in_world_rot_mat)
                    #print("----------")
                    #print("Angle j1: ", angle_j1)
                if show_left_arm_j2:
                    angle_j2 = calculate_angle_j2(shoulder_coords_in_world_rot_mat)
                    #print("----------")
                    #print("Angle j2: ", angle_j2)

            if show_left_arm_j3 or show_left_arm_j4:
                x_elbow, y_elbow, z_elbow = elbow_coordinate_wrt_origin[:, 0], elbow_coordinate_wrt_origin[:, 1], elbow_coordinate_wrt_origin[:, 2]
                elbow_coordinate_wrt_shoulder = calculate_elbow_coordinate_wrt_shoulder(shoulder_coords_in_world,
                    elbow_coordinate_wrt_origin)  # (3, O), O = number of vectors (xyz)
                elbow_coordinate_wrt_shoulder_rot_mat = R.from_matrix(elbow_coordinate_wrt_shoulder)               

                pts = np.concatenate([pts, [x_elbow * 40, y_elbow * 40, z_elbow * 40]], axis=0)
                last_index = pts.shape[0] - 1
                lines.extend([[0, last_index - 2], [0, last_index - 1], [0, last_index]])
                colors.extend([(1, 0, 0), (0, 1, 0), (0, 0, 1)])

                if show_left_arm_j3:
                    angle_j3 = calculate_angle_j3(elbow_coordinate_wrt_shoulder_rot_mat)
                    #print("----------")
                    #print("Angle j3: ", angle_j3)
                if show_left_arm_j4:
                    angle_j4 = calculate_angle_j4(elbow_coordinate_wrt_shoulder_rot_mat)
                    #print("----------")
                    #print("Angle j4: ", angle_j4)

            if show_left_arm_j5 or show_left_arm_j6:
                x_wrist, y_wrist, z_wrist = wrist_coordinate_wrt_origin[:, 0], wrist_coordinate_wrt_origin[:, 1], wrist_coordinate_wrt_origin[:, 2]
                wrist_coordinate_wrt_elbow = calculate_wrist_coordinate_wrt_elbow(elbow_coordinate_wrt_origin,
                    wrist_coordinate_wrt_origin)  # (3, O), O = number of vectors (xyz)
                wrist_coordinate_wrt_elbow_rot_mat = R.from_matrix(wrist_coordinate_wrt_elbow)

                pts = np.concatenate([pts, [x_wrist * 40, y_wrist * 40, z_wrist * 40]], axis=0)
                last_index = pts.shape[0] - 1
                lines.extend([[0, last_index - 2], [0, last_index - 1], [0, last_index]])
                colors.extend([(1, 0, 0), (0, 1, 0), (0, 0, 1)])

                if show_left_arm_j5:
                    angle_j5 = calculate_angle_j5(wrist_coordinate_wrt_elbow_rot_mat, wrist_coordinate_wrt_elbow)
                    print("----------")
                    print("Angle j5: ", angle_j5)
                if show_left_arm_j6:
                    angle_j6 = calculate_angle_j6(wrist_coordinate_wrt_elbow_rot_mat)
                    print("----------")
                    print("Angle j6: ", angle_j6)
            
            pcd.points = o3d.utility.Vector3dVector(pts)
            line_set.points = o3d.utility.Vector3dVector(pts)  # Update the points
            line_set.lines = o3d.utility.Vector2iVector(lines)  # Update the lines
            line_set.colors = o3d.utility.Vector3dVector(colors)

            # Draw cuboid
            min_x, min_y, min_z = np.min(pts, axis=0)
            max_x, max_y, max_z = np.max(pts, axis=0)
            vertices = [
                [min_x, min_y, min_z], [min_x, min_y, max_z], [min_x, max_y, min_z], [min_x, max_y, max_z],
                [max_x, min_y, min_z], [max_x, min_y, max_z], [max_x, max_y, min_z], [max_x, max_y, max_z]
            ]
            vertices = np.array(vertices) 
            edges = [
                [0, 1], [1, 3], [3, 2], [2, 0],  
                [4, 5], [5, 7], [7, 6], [6, 4],  
                [0, 4], [1, 5], [2, 6], [3, 7]   
            ]
            bounding_box.points = o3d.utility.Vector3dVector(vertices)
            bounding_box.lines = o3d.utility.Vector2iVector(edges)
            bbox_colors = [[0, 1, 0] for _ in range(len(edges))]  
            bounding_box.colors = o3d.utility.Vector3dVector(bbox_colors)

            # Update the visualization
            vis.update_geometry(pcd)
            vis.update_geometry(line_set)
            vis.update_geometry(bounding_box)
        vis.poll_events()
        vis.update_renderer()        

        time.sleep(0.1)  # Set time sleep here is importance, the higher time.sleep parameter is (unit is second), the faster the main thread can process

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