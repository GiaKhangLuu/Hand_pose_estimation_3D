import numpy as np
import open3d as o3d
import threading
import queue
import time
import random
from scipy.spatial.transform import Rotation as R
#my_queue = queue.Queue()

def visualization_thread(lmks_queue, draw_xyz=True):
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
            #wrist, finger_lmks = lmks_queue.get()
            #finger_lmks = np.reshape(finger_lmks, (-1, 3))
            #pts = np.vstack((wrist[None, :], finger_lmks))
            pts = lmks_queue.get()
            lines = [[0, 1], [1, 2], [2, 3], [2, 4], [2, 5], [3, 4],
                     [0, 6], [0, 7],
                     [6, 8], [7, 8]]
            colors = [[0, 0, 0] for i in range(len(lines))]

            if draw_xyz:
                #pts = np.concatenate([pts, [[20, 0, 0], [0, 20, 0], [0, 0, 20]]], axis=0)
                #lines.extend([[0, 9], [0, 10], [0, 11]])
                #colors.extend([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

                # Debugging calculating joint 3, will remove the following lines after finishing
                landmark_dictionary = ["left shoulder", "left elbow", "left wrist",
                                       "left pinky", "left index", "left thumb", "left hip",
                                       "right shoulder", "right hip"]
                angle_1 = calculate_angle_j1(pts)
                angle_2 = cal_angle2(pts)
                rot_mat = R.from_euler("yz", [angle_2, angle_1], degrees=True).as_matrix()

                z_new = np.matmul(rot_mat, [0, 0, 1])
                y_new = pts[landmark_dictionary.index("left elbow")]
                x_new = np.cross(y_new, z_new)

                x_new = x_new / np.linalg.norm(x_new)
                y_new = y_new / np.linalg.norm(y_new)
                z_new = z_new / np.linalg.norm(z_new)

                trans_mat = np.array([x_new, y_new, z_new])
                trans_mat = np.transpose(trans_mat)
                trans_mat_inv = np.linalg.inv(trans_mat)

                b = pts[landmark_dictionary.index("left wrist")] - pts[landmark_dictionary.index("left elbow")]
                b_prime = np.matmul(trans_mat_inv, b.T)
                b_prime = b_prime.T
                b_prime_proj = b_prime * [1, 1, 0]

                b_prime_in_world_to_plot = np.matmul(trans_mat, b_prime.T)
                b_prime_in_world_to_plot = b_prime_in_world_to_plot.T
                b_prime_proj_in_world_to_plot = np.matmul(trans_mat, b_prime_proj.T)
                b_prime_proj_in_world_to_plot = b_prime_proj_in_world_to_plot.T

                pts = np.concatenate([pts, [b_prime_in_world_to_plot, b_prime_proj_in_world_to_plot, z_new * 20]], axis=0)
                lines.extend([[0, 9], [0, 10], [0, 11]])
                colors.extend([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

            pcd.points = o3d.utility.Vector3dVector(pts)
            line_set.points = o3d.utility.Vector3dVector(pts)  # Update the points
            line_set.lines = o3d.utility.Vector2iVector(lines)  # Update the lines
            line_set.colors = o3d.utility.Vector3dVector(colors)

            # Update the visualization
            vis.update_geometry(pcd)
            vis.update_geometry(line_set)
        vis.poll_events()
        vis.update_renderer()        

        time.sleep(0.01)

    vis.destroy_window()

def cal_angle2(XYZ_landmarks):
    a = np.array([1, 0, 0])
    b = XYZ_landmarks[1]
    b = b * [1, 0, 1]

    dot = np.sum(a * b)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = dot / (a_norm * b_norm)
    angle = np.degrees(np.arccos(cos))
    angle = 180 - angle
    c = np.cross(b, a)
    ref = c[1] + 1e-9
    signs = ref / np.absolute(ref)
    angle *= signs
    return angle

def calculate_angle_j1(XYZ_landmarks):
    a = np.array([1, 0, 0])
    b = XYZ_landmarks[1]
    b = b * [1, 1, 0]

    dot = np.sum(a * b)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = dot / (a_norm * b_norm)
    angle_j1 = np.degrees(np.arccos(cos))

    angle_j1 = 180 - angle_j1

    c = np.cross(b, a)
    ref = c[-1] + 1e-9
    signs = ref / np.absolute(ref)

    angle_j1 *= signs

    return angle_j1

def add_to_queue(my_queue):
    random_number = random.random()
    if random_number > 0.5:
        wrist = np.array([2.84217094e-14, -5.68434189e-14, -1.42108547e-14])
        fingers = np.array([[[1.81827529e+01, 2.70208977e+01,  1.56907294e+01],
                             [2.87178244e+01, 5.39137715e+01,  1.91041062e+01],
                             [4.16239651e+01, 7.80126691e+01,  1.22261161e+01],
                             [5.12960969e+01, 9.44579794e+01,  5.09180066e+00]],
                            [[2.84217094e-14, 8.70890376e+01,  1.79453478e+01],
                             [2.72279716e+01, 1.06530983e+02,  1.14512460e+01],
                             [3.49863673e+01, 9.01106687e+01,  1.11114118e+01],
                             [3.22039925e+01, 7.78760866e+01,  1.25123744e+01]],
                            [[2.84217094e-14, 8.56721993e+01, -1.42108547e-14],
                             [2.79436557e+01, 1.07582576e+02, -4.30093475e+00],
                             [3.59293720e+01, 8.71054896e+01, -1.84081791e+00],
                             [3.22374695e+01, 7.55243499e+01,  5.85769381e-01]],
                            [[3.25899110e+00, 7.92462728e+01, -1.49791245e+01],
                             [2.97091493e+01, 9.64890772e+01, -1.70255333e+01],
                             [3.60970207e+01, 7.81485364e+01, -1.36189556e+01],
                             [3.02891575e+01, 6.74966579e+01, -1.08961284e+01]],
                            [[7.85462810e+00, 6.84820196e+01, -2.59124205e+01],
                             [3.03425025e+01, 8.33986698e+01, -2.73843180e+01],
                             [3.46067830e+01, 7.12892082e+01, -2.32612341e+01],
                             [2.83563686e+01, 6.22487689e+01, -1.98816931e+01]]])
    else:
        wrist = np.array([0.00000000e+00, 0.00000000e+00, -7.10542736e-15])
        fingers = np.array([[[2.06227723e+01, 2.81387400e+01, 1.40803731e+01],
                             [3.13269680e+01, 5.56275331e+01, 1.71974038e+01],
                             [4.26650009e+01, 7.93375730e+01, 1.05669728e+01],
                             [5.15103885e+01, 9.55016045e+01, 3.73768733e+00]],
                            [[2.84217094e-14, 8.75200661e+01, 1.83381066e+01],
                             [2.74545481e+01, 1.07455359e+02, 1.03489977e+01],
                             [3.53065736e+01, 9.11815770e+01, 1.03330300e+01],
                             [3.26603371e+01, 7.93154391e+01, 1.23555513e+01]],
                            [[0.00000000e+00, 8.63661187e+01,-7.10542736e-15],
                             [2.79793797e+01, 1.08099253e+02,-5.85302375e+00],
                             [3.57728408e+01, 8.78444458e+01,-3.70866694e+00],
                             [3.12276870e+01, 7.70233197e+01,-1.23803140e+00]],
                            [[3.31056722e+00, 8.04254807e+01,-1.56162384e+01],
                             [2.98843534e+01, 9.73543170e+01,-1.87477107e+01],
                             [3.55651813e+01, 7.93217289e+01,-1.57529277e+01],
                             [2.87912845e+01, 6.93763786e+01,-1.30955968e+01]],
                            [[8.17773306e+00, 7.07002037e+01,-2.74413320e+01],
                             [2.98932889e+01, 8.44776747e+01,-2.88192214e+01],
                             [3.35218111e+01, 7.27390844e+01,-2.48312620e+01],
                             [2.64891304e+01, 6.45920956e+01,-2.16830720e+01]]])

    my_queue.put((wrist, fingers))                    

if __name__ == "__main__":
    my_queue = queue.Queue() 
    vis_thread = threading.Thread(target=visualization_thread, args=(my_queue,), daemon=True)
    vis_thread.start()

    while True:
        add_to_queue(my_queue)
        time.sleep(1)

