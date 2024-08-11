import cv2
import depthai as dai
import numpy as np
import pyrealsense2 as rs
import threading
import queue
import os
import shutil

def calibrate(cam, is_found_in_both_cams, showPics=True):
    # Read Image
    root = os.getcwd()
    calibrationDir = os.path.join(root, 'images', cam)
    #imgPathList = glob.glob(os.path.join(calibrationDir, '*.png'))
    imgPathList = os.listdir(calibrationDir)

    # Filter out only .png files (optional, in case there are other types of files in the directory)
    png_files = [f for f in imgPathList if f.endswith('.png')]

    # Sort the filenames numerically
    imgPathList = sorted(png_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    ## Initialize
    nRows = 11
    nCols = 8
    termCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    worldPtsCur = np.zeros((nRows*nCols,3), np.float32)
    worldPtsCur[:,:2] = np.mgrid[0:nRows,0:nCols].T.reshape(-1,2)
    worldPtsCur *= 15 
    #print(worldPtsCur)
    worldPtsList = []
    imgPtsList = []

    assert len(imgPathList) == len(is_found_in_both_cams)

    # Find Corners
    for i, curImgPath in enumerate(imgPathList):
        if is_found_in_both_cams[i]:
            print(curImgPath)
            imgBGR = cv2.imread(os.path.join(calibrationDir, curImgPath))
            imgGray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
            imgGray = cv2.equalizeHist(imgGray)
            cornersFound, cornersOrg = cv2.findChessboardCorners(imgGray, (nRows, nCols), None)

            if cornersFound == True:
                worldPtsList.append(worldPtsCur)
                cornersRefined = cv2.cornerSubPix(imgGray, cornersOrg, (11,11), (-1,-1), termCriteria) 
                imgPtsList.append(cornersRefined)
                if showPics:
                    cv2.drawChessboardCorners(imgBGR, (nRows, nCols), cornersRefined, cornersFound)
                    cv2.imshow("Chessboard", imgBGR)
                    cv2.waitKey(500)
    cv2.destroyAllWindows()

    #print("Num found of {}: {}".format(cam, num_found))

    ## Calibrate
    repError, camMatrix, distCoeff, rvecs, tvecs = cv2.calibrateCamera(worldPtsList, imgPtsList, imgGray.shape[::-1], None, None)
    print('Camera Matrix: \n', camMatrix)
    print("Reproj Error (pixels): {:.4f}".format(repError))

    # Save Calibration Parameters (later video)
    curFolder = os.path.dirname(os.path.abspath(__file__))
    paramPath = os.path.join(curFolder, '{}_calibration.npz'.format(cam))
    np.savez(paramPath,
            repError=repError,
            camMatrix=camMatrix, 
            distCoeff=distCoeff,
            rvecs=rvecs,
            tvecs=tvecs)

    return camMatrix

def runCalibration(cam, is_found_in_both_cams):
    calibrate(cam, is_found_in_both_cams, showPics=True)

# OAK-D processing function
def process_oak(oak_queue, mxid=None):
    pipeline_oak = dai.Pipeline()

    cam_rgb = pipeline_oak.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setFps(30)

    xout_rgb = pipeline_oak.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.video.link(xout_rgb.input)

    if mxid is not None:
        device_info = dai.DeviceInfo(mxid) 
        device_oak = dai.Device(pipeline_oak, device_info)
    else:
        device_oak = dai.Device(pipeline_oak)
    rgb_queue_oak = device_oak.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        rgb_frame_oak = rgb_queue_oak.get()

        frame_oak = rgb_frame_oak.getCvFrame()

        frame_oak = cv2.resize(frame_oak, (1280, 720))
        #depth_oak_display = cv2.resize(depth_oak, (640, 480))

        oak_queue.put(frame_oak)

        if oak_queue.qsize() > 1:
            oak_queue.get()

oak_queue = queue.Queue(maxsize=1)

# Start RealSense and OAK-D processing threads
oak_mxid = '18443010613E940F00'
oak_thread = threading.Thread(target=process_oak, args=(oak_queue, oak_mxid))

oak_thread.start()

folder_path = './images'
oak_path = './images/oak'

if os.path.exists(folder_path):
    shutil.rmtree(folder_path)
os.makedirs(folder_path)

if not os.path.exists(oak_path):
    os.makedirs(oak_path)

num = 0

# Capture image
while True:

    #if (not rs_queue.empty()) and (not oak_queue.empty()):
    if not oak_queue.empty():
        frame = oak_queue.get()
        cv2.imshow("OAK Frame", frame)

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('{}/oak_img_'.format(oak_path) + str(num) + '.png', frame)
        print("image saved!")
        num += 1
    elif k == ord("q"):
        cv2.destroyAllWindows()
        break

# Start calibrating
## Initialize
nRows = 11
nCols = 8
termCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
masks = []
    
for i, cam in enumerate(['oak']):
    #root = os.getcwd()
    #calibrationDir = os.path.join(root, 'images', cam)
    #runCalibration(cam)
    root = os.getcwd()
    calibrationDir = os.path.join(root, 'images', cam)
    #imgPathList = glob.glob(os.path.join(calibrationDir, '*.png'))
    imgPathList = os.listdir(calibrationDir)

    # Filter out only .png files (optional, in case there are other types of files in the directory)
    png_files = [f for f in imgPathList if f.endswith('.png')]

    # Sort the filenames numerically
    imgPathList = sorted(png_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    is_find_corners = []

    # Find corners
    for curImgPath in imgPathList:
        curImgPath = os.path.join(calibrationDir, curImgPath)
        imgBGR = cv2.imread(curImgPath)
        imgGray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
        imgGray = cv2.equalizeHist(imgGray)
        cornersFound, cornersOrg = cv2.findChessboardCorners(imgGray, (nRows, nCols), None)

        is_find_corners.append(cornersFound)

    masks.append(is_find_corners)

#masks = (np.array(masks[0]) & np.array(masks[1])).tolist()
masks = np.array(masks[0]).tolist()

for i, cam in enumerate(['oak']):
    runCalibration(cam, masks)