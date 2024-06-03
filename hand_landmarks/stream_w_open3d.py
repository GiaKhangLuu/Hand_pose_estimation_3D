import numpy as np
import open3d as o3d
import threading
import queue
import time

my_queue = queue.Queue()

def visualization_thread():
    # Generate random fingers and wrist
    """
    fingers = np.random.rand(20, 3)
    wrist = np.array([2.84217094e-14, 5.68434189e-14, 1.42108547e-14])
    pts = np.vstack((wrist.reshape(1, -1), fingers))

    # Create the initial point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    """
    
    x = np.array([[1, 0, 0],
                  [0, 0, 0]])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(x)

    lines = [[0, 1]]
    colors = [[1, 0, 0] for i in range(len(lines))]
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
        if not my_queue.empty():
            #points = my_queue.get()
            #fingers = np.random.rand(20, 3)
            #wrist = np.array([2.84217094e-14, 5.68434189e-14, 1.42108547e-14])
            #pts = np.vstack((wrist.reshape(1, -1), fingers))
            ##pts = my_queue.get()

            # Create the initial point cloud
            pts = my_queue.get()
            pcd.points = o3d.utility.Vector3dVector(pts)

            lines = [[0,1],[1,2],[2,3],[3,4], 
                     [0,5],[5,6],[6,7],[7,8],
                     [5,9],[9,10],[10,11],[11,12],
                     [9,13],[13,14],[14,15],[15,16],
                     [13,17],[17,18],[18,19],[19,20],[0,17]]
            colors = [[1, 0, 0] for i in range(len(lines))]
            line_set.points = o3d.utility.Vector3dVector(pts)  # Update the points
            line_set.lines = o3d.utility.Vector2iVector(lines)  # Update the lines
            line_set.colors = o3d.utility.Vector3dVector(colors)

            # Update the visualization
            vis.update_geometry(pcd)
            vis.update_geometry(line_set)
            vis.poll_events()
            vis.update_renderer()
        
        time.sleep(0.001)

    vis.destroy_window()

# Start the visualization thread
vis_thread = threading.Thread(target=visualization_thread)

def add_to_queue():
    fingers = np.random.rand(20, 3)
    wrist = np.array([2.84217094e-14, 5.68434189e-14, 1.42108547e-14])
    pts = np.vstack((wrist.reshape(1, -1), fingers))
    my_queue.put(pts)

if __name__ == "__main__":
    vis_thread.start()

    while True:
        add_to_queue()
        time.sleep(0.01)