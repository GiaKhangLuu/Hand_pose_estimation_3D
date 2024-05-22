import threading

x = []

def add():
    x.append(5)

thread = threading.Thread(target=process_mediapipe, args=(rs_queue, frame_manager[0], "RealSense"))