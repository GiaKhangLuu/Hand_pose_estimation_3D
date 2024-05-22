import numpy as np
import cv2 
import glob
import os
import matplotlib.pyplot as plt

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

if __name__ == "__main__":

    ## Initialize
    nRows = 11
    nCols = 8
    termCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    masks = []
    
    for i, cam in enumerate(['rs', 'oak']):
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
            cornersFound, cornersOrg = cv2.findChessboardCorners(imgGray, (nRows, nCols), None)

            is_find_corners.append(cornersFound)

        masks.append(is_find_corners)

    masks = (np.array(masks[0]) & np.array(masks[1])).tolist()

    for i, cam in enumerate(['rs', 'oak']):
        runCalibration(cam, masks)

