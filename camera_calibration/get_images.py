import cv2
import os
import time

cap = cv2.VideoCapture(0)

num = 0

folder_path = './images'
rs_path = './images/rs'
oak_path = './images/oak'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

if not os.path.exists(rs_path):
    os.makedirs(rs_path)

if not os.path.exists(oak_path):
    os.makedirs(oak_path)

# Get the initial time
start_time = time.time()

while cap.isOpened():
    # Read a frame from the camera
    success, img = cap.read()

    # Check if frame reading was successful
    if not success:
        print("Failed to capture image")
        break

    # Display the captured image
    cv2.imshow('Img', img)

    # Check the elapsed time
    elapsed_time = time.time() - start_time

    # If 2 seconds have passed, save the current frame as an image
    if elapsed_time >= 2:
        # Save the image with a unique name
        image_path = os.path.join(folder_path, f'img{num}.png')
        cv2.imwrite(image_path, img)
        print(f"Image saved as {image_path}")
        # Increment the image counter
        num += 1
        # Reset the start time
        start_time = time.time()

    # Wait for a key press for 1 millisecond
    k = cv2.waitKey(1)

    # If 'Esc' key (ASCII value 27) is pressed, exit the loop
    if k == 27:
        break

# Release the camera and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()