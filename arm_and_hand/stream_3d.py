import sys
import os

#CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
#sys.path.append(os.path.join(CURRENT_DIR, '..'))

import numpy as np
import open3d as o3d
import threading
import time
from scipy.spatial.transform import Rotation as R

def visualize_arm(lmks_queue,
    landmark_dictionary, 
    left_arm_angle_calculator=None,
    left_hand_angle_calculator=None,
    draw_original_xyz=True,
    draw_left_arm_coordinates=None,
    draw_left_hand_coordinates=None,
    joint_vector_color=None,
    ref_vector_color=None):
    """
    TODO: Doc.
    """

    x = np.array([[0, 0, 0],
                  [500, 0, 0],
                  [0, 500, 0],
                  [0, 0, 500]])
    SCALE_FACTOR = 50
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

    #landmark_dictionary = landmark_dictionary[5:]
    
    while True:
        """
        -> Coordinate of a FIRST JOINT in a `chain` always rotates about the z-axis
            of a LAST JOINT  in a `previous chain`. Therefore, when plotting, we 
            can see that they share the same z-axis.
        -> Coordinate of a LAST JOINT in a `chain` always rotatees about the y-axis
            of a PREVIOUS JOINT in a `same chain`. Therefore, when plotting, we 
            can see that thay share the same y-axis.
        """
        if not lmks_queue.empty():
            pts, original_xyz, left_arm_result, left_hand_result = lmks_queue.get()  # pts.shape = (M, N), M = #vectors = 21, N = #features = 3. original_xyz.shape = (N, O), N = #features = 3, O = #vectors = 3 (xyz)
            lines = [[0, 1], [1, 5], [0, 2], [0, 3], [2, 4], 
                [3, 4], [5, 6], [5, 10], [14, 18], [18, 22], 
                [10, 14], [5, 22], [6, 7], [7, 8], [8, 9], 
                [10, 11], [11, 12], [12, 13], [14, 15], [15, 16], 
                [16, 17], [18, 19], [19, 20], [20, 21], [22, 23], 
                [23, 24], [24, 25]]

            # Only show hand
            #pts = pts[5:]
            #lines = [[0, 1], [0, 5], [9, 13], [13, 17],
                #[5, 9], [0, 17], [1, 2], [2, 3], [3, 4],
                #[5, 6], [6, 7], [7, 8], [9, 10], [10, 11],
                #[11, 12], [13, 14], [14, 15], [15, 16], [17, 18],
                #[18, 19], [19, 20]]

            colors = [[0, 0, 0] for i in range(len(lines))]

            if draw_original_xyz:
                assert original_xyz is not None
                x_unit, y_unit, z_unit = original_xyz[:, 0], original_xyz[:, 1], original_xyz[:, 2]
                pts = np.concatenate([pts, [x_unit * 40, y_unit * 40, z_unit * 40]], axis=0)
                last_index = pts.shape[0] - 1
                lines.extend([[0, last_index - 2], [0, last_index - 1], [0, last_index]])
                colors.extend([(1, 0, 0), (0, 1, 0), (0, 0, 1)])

            left_arm_angles = left_arm_result["left_arm"]["angles"]
            left_arm_rot_mats_wrt_origin = left_arm_result["left_arm"]["rot_mats_wrt_origin"]
            parent_coordinate = original_xyz

            for landmark_idx, landmark_name in enumerate(left_arm_angle_calculator.landmarks_name):
                for angle_idx_in_chain in range(left_arm_angle_calculator.num_angles_each_chain):
                    if landmark_name in ["shoulder", "elbow"]:
                        landmark_name_in_dictionary = f"left {landmark_name}"
                    else:
                        landmark_name_in_dictionary = landmark_name
                    joint_idx = landmark_idx * left_arm_angle_calculator.num_angles_each_chain + angle_idx_in_chain
                    ji_idx = landmark_dictionary.index(landmark_name_in_dictionary)
                    ji_landmark = pts[ji_idx].copy()

                    left_arm_angles_ji = left_arm_angles[joint_idx]
                    left_arm_ji_rot_mat_wrt_origin = left_arm_rot_mats_wrt_origin[joint_idx]

                    left_arm_x_ji = left_arm_ji_rot_mat_wrt_origin[:, 0] * SCALE_FACTOR + ji_landmark
                    left_arm_y_ji = left_arm_ji_rot_mat_wrt_origin[:, 1] * SCALE_FACTOR + ji_landmark
                    left_arm_z_ji = left_arm_ji_rot_mat_wrt_origin[:, 2] * SCALE_FACTOR + ji_landmark
                    
                    if angle_idx_in_chain == 0:
                        parent_coordinate_prime = parent_coordinate @ left_arm_angle_calculator.rot_mat_to_rearrange_container[landmark_idx] 
                        left_arm_x_ji_parent = parent_coordinate_prime[:, 0] * SCALE_FACTOR + ji_landmark 
                        left_arm_y_ji_parent = parent_coordinate_prime[:, 1] * SCALE_FACTOR + ji_landmark 
                        left_arm_z_ji_parent = parent_coordinate_prime[:, 2] * SCALE_FACTOR + ji_landmark 

                    if angle_idx_in_chain == left_arm_angle_calculator.num_angles_each_chain - 1:
                        parent_coordinate = left_arm_ji_rot_mat_wrt_origin 

                    if draw_left_arm_coordinates[f"show_left_arm_joint{joint_idx+1}"]:
                        pts = np.concatenate([pts, [left_arm_x_ji, left_arm_y_ji, left_arm_z_ji]], axis=0)
                        last_index = pts.shape[0] - 1
                        lines.extend([[ji_idx, last_index - 2], [ji_idx, last_index - 1], [ji_idx, last_index]])
                        colors.extend([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

                        if (draw_left_arm_coordinates["show_parent_coordinate"] and
                            angle_idx_in_chain == 0):
                            draw_left_arm_coordinates[f"show_left_arm_joint{joint_idx+2}"] = False
                            pts = np.concatenate([pts, [left_arm_x_ji_parent, left_arm_y_ji_parent, left_arm_z_ji_parent]], axis=0)
                            last_index = pts.shape[0] - 1
                            lines.extend([[ji_idx, last_index - 2], [ji_idx, last_index - 1], [ji_idx, last_index]])
                            colors.extend([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

                        print("----------------")
                        print(f"Angle j{joint_idx+1}: {round(left_arm_angles_ji)}")

            for finger_idx, finger_name in enumerate(left_hand_angle_calculator.fingers_name):
                parent_coordinate = left_arm_rot_mats_wrt_origin[-1]
                left_finger_angle_calculator = left_hand_angle_calculator.fingers_calculator[finger_idx]
                left_finger_angles = left_hand_result[finger_name]["angles"]
                left_finger_rot_mats_wrt_origin = left_hand_result[finger_name]["rot_mats_wrt_origin"]

                for landmark_idx, landmark_name in enumerate(left_finger_angle_calculator.landmarks_name):
                    for angle_idx_in_chain in range(left_finger_angle_calculator.num_angles_each_chain):
                        if finger_name in ["INDEX", "MIDDLE", "RING"]:
                            landmark_name_in_dictionary = f"{finger_name}_FINGER_{landmark_name}"
                        else:
                            landmark_name_in_dictionary = f"{finger_name}_{landmark_name}"

                        joint_idx = landmark_idx * left_finger_angle_calculator.num_angles_each_chain + angle_idx_in_chain
                        ji_idx = landmark_dictionary.index(landmark_name_in_dictionary)
                        ji_landmark = pts[ji_idx].copy()

                        if (angle_idx_in_chain == left_finger_angle_calculator.num_angles_each_chain - 1 and
                            not left_finger_angle_calculator.calculate_second_angle_flag_container[landmark_idx]):
                            break

                        left_finger_angles_ji = left_finger_angles[joint_idx]
                        left_finger_ji_rot_mat_wrt_orgin = left_finger_rot_mats_wrt_origin[joint_idx]

                        left_finger_x_ji = left_finger_ji_rot_mat_wrt_orgin[:, 0] * SCALE_FACTOR + ji_landmark
                        left_finger_y_ji = left_finger_ji_rot_mat_wrt_orgin[:, 1] * SCALE_FACTOR + ji_landmark
                        left_finger_z_ji = left_finger_ji_rot_mat_wrt_orgin[:, 2] * SCALE_FACTOR + ji_landmark

                        if angle_idx_in_chain == 0:
                            parent_coordinate_prime = parent_coordinate @ left_finger_angle_calculator.rot_mat_to_rearrange_container[landmark_idx]
                            left_finger_x_ji_parent = parent_coordinate_prime[:, 0] * SCALE_FACTOR + ji_landmark
                            left_finger_y_ji_parent = parent_coordinate_prime[:, 1] * SCALE_FACTOR + ji_landmark
                            left_finger_z_ji_parent = parent_coordinate_prime[:, 2] * SCALE_FACTOR + ji_landmark

                        if angle_idx_in_chain == left_finger_angle_calculator.num_angles_each_chain - 1:
                            parent_coordinate = left_finger_ji_rot_mat_wrt_orgin

                        if draw_left_hand_coordinates[finger_name][f"show_left_finger_joint{joint_idx+1}"]:
                            pts = np.concatenate([pts, [left_finger_x_ji, left_finger_y_ji, left_finger_z_ji]], axis=0)
                            last_index = pts.shape[0] - 1
                            lines.extend([[ji_idx, last_index - 2], [ji_idx, last_index - 1], [ji_idx, last_index]])
                            colors.extend([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

                            if (draw_left_hand_coordinates[finger_name]["show_parent_coordinate"] and
                                angle_idx_in_chain == 0):
                                draw_left_hand_coordinates[finger_name][f"show_left_finger_joint{joint_idx+2}"] = False
                                pts = np.concatenate([pts, [left_finger_x_ji_parent, left_finger_y_ji_parent, left_finger_z_ji_parent]], axis=0)
                                last_index = pts.shape[0] - 1
                                lines.extend([[ji_idx, last_index - 2], [ji_idx, last_index - 1], [ji_idx, last_index]])
                                colors.extend([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

                            print("----------------")
                            print(f"Angle of {finger_name} j{joint_idx+1}: {round(left_finger_angles_ji)}")

            pcd.points = o3d.utility.Vector3dVector(pts)
            line_set.points = o3d.utility.Vector3dVector(pts)  
            line_set.lines = o3d.utility.Vector2iVector(lines)  
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

            vis.update_geometry(pcd)
            vis.update_geometry(line_set)
            vis.update_geometry(bounding_box)
        vis.poll_events()
        vis.update_renderer()        

        time.sleep(0.1)  # Set time sleep here is importance, the higher time.sleep parameter is (unit is second), the faster the main thread can process

    vis.destroy_window()
