import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
 
cv.namedWindow("tracking")
vc = cv.VideoCapture(0)
 
if vc.isOpened():
    ret, frame = vc.read()
else:
    rval = False

# Random globals ripped off internet
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


# OUR STUFF
# Set up blob detection
params = cv.SimpleBlobDetector_Params()

# Set parameters for blob detection
params.filterByArea = True
params.minArea = 100
params.maxArea = 1000

params.filterByCircularity = False
params.minCircularity = 0.7

params.filterByConvexity = True
params.minConvexity = 0.8

params.filterByInertia = True
params.minInertiaRatio = 0.5

params.filterByColor = True
params.blobColor = 0

detector = cv.SimpleBlobDetector_create(params)

# detector = cv.SimpleBlobDetector()

while True:
    ret, frame = vc.read() # Read a frame
    if not ret:
        print("Error: Could not read frame.")
        break
        
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # cv.imshow('Camera Feed', frame) # Display the frame
    
    if cv.waitKey(1) & 0xFF == ord('q'): # Exit on 'q' key press
        break

    
    
    chessboardPresent, corners = cv.findChessboardCorners(image = frame, patternSize = (9, 6))
    if (chessboardPresent):
        validCorners = corners

     # If found, add object points, image points (after refining them)
    if chessboardPresent == True:
        objpoints.append(objp)
 
        corners2 = cv.cornerSubPix(frame, corners, (11,11), (-1,-1), criteria)
        # imgpoints.append(corners2)
 
        # Draw and display the corners
        cv.drawChessboardCorners(frame, (9,6), corners2, ret)
        # cv.drawChessboardCorners(frame, (9,6), corners, ret)
        # cv.imshow('Chessboard Corners', frame)
        cv.imshow("Cup detection", frame)
    else:
        keypoints = detector.detect(frame)
        im_with_keypoints = cv.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imshow("Cup detection", im_with_keypoints)
        # if (keypoints):
        #     keypoint = max(keypoints, key=lambda p: p.size)
        #     # Draw only the largest keypoint
        #     im_with_keypoints = cv.drawKeypoints(frame,[keypoint], np.array([]), color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        #     # cv.imshow("Blob Detection", im_with_keypoints)
        #     cv.imshow("Cup detection", im_with_keypoints)
        # else:
        #     cv.imshow("Cup detection", frame)
        
        for keypoint in keypoints:
            kp_x, kp_y = keypoint.pt
        
            # Check if the blob's center is between adjacent corners
            for i in range(len(validCorners)-1):
                # Get two consecutive validCorners (you can adjust this for grid alignment)
                corner1 = validCorners[i][0]
                corner2 = validCorners[i+1][0]
                
                # Create a bounding box between the two validCorners (for simplicity, we use axis-aligned bounding box)
                min_x, max_x = min(corner1[0], corner2[0]), max(corner1[0], corner2[0])
                min_y, max_y = min(corner1[1], corner2[1]), max(corner1[1], corner2[1])

                # Check if the keypoint's center lies within this bounding box
                if min_x <= kp_x <= max_x and min_y <= kp_y <= max_y:
                    # Draw a rectangle around the bounding box where the blob is located
                    # cv.rectangle(im_with_keypoints, (int(min_x), int(min_y)), (int(max_x), int(max_y)), color = (0, 255, 0), thickness=2)
                    cv.circle(im_with_keypoints, (int(kp_x), int(kp_y)), 5, (0, 0, 255), -1)  # Mark the blob center with a red circle
                    cv.imshow("Cup detection", im_with_keypoints)


    cv.waitKey(50)

    

# while ret:
    
    
#     # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     # cv.imshow("tracking", frame, cmap='gray')
#     # cv.imshow("tracking", frame)
#     chessboardPresent, corners = cv.findChessboardCorners(image = frame, patternSize = (9, 6))
#     corners = np.squeeze(corners)
#     img = np.copy(frame)
#     for corner in corners:
#         coord = (int(corner[0]), int(corner[1]))
   
#     obj_grid = np.zeros((12*5,3), np.float32)
#     obj_grid[:,:2] = np.mgrid[0:12,0:6].T.reshape(-1,2)
#     rval, frame = vc.read()
#     break
#     # key = cv.waitKey(20)
#     # if key == 27:
#     #     break
 


 
cv.destroyWindow("tracking")
vc.release()