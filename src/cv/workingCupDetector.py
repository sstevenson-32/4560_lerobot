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


# OUR STUFF
board_size = (9, 6) 
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
        corners2 = cv.cornerSubPix(frame, corners, (11,11), (-1,-1), criteria)
        
        # Draw and display the corners
        cv.drawChessboardCorners(frame, (9,6), corners2, ret)
        cv.imshow("Cup detection", frame)
    else:
        keypoints = detector.detect(frame)
        im_with_keypoints = cv.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv.imshow("Cup detection", im_with_keypoints)
        
        # Convert corners to a 2D numpy array (corners is a list of points)
        corners_2d = np.array(corners2, dtype=np.float32)
        
        # Reshape the array into the same dimensions as the chessboard grid (rows, columns)
        # (Note: the size of the chessboard is (9,6), so we will reshape it into (6, 9))
        corners_grid = corners_2d.reshape(board_size[1], board_size[0], 2)  # (rows, columns, [x, y])
        
        firstCornerFound = False
        firstCorner = [-1,-1]

        # Check the first column (this is outside of the innter corners)
        for row in range(len(corners_grid) - 1):  # Iterate through through each of 6 rows
            for col in range(len(corners_grid[row]) - 1): # Iterate through each of 9 corners per row 
                if ((row + col) % 2 == 0):
                    corner_tl = corners_grid[row][col]
                    corner_tr = corners_grid[row][col + 1]
                    corner_bl = corners_grid[row + 1][col]
                    corner_br = corners_grid[row + 1][col + 1]

                    boxFilled = False

                    for keypoint in keypoints:
                        kp_x, kp_y = keypoint.pt

                        # Check if the keypoint's center lies within this bounding box
                        if corner_tr[0] <= kp_x <= corner_tl[0] and corner_br[1] <= kp_y <= corner_tl[1]:
                            boxFilled = True
                            cv.circle(im_with_keypoints, (int(kp_x), int(kp_y)), 5, (0, 0, 255), -1)  # Mark the blob center with a red circle
                            
                    if (not boxFilled):
                        if (not firstCornerFound):
                            firstCorner = [row, col]
                            firstCornerFound = True
                            print(firstCorner)
                            print(corner_tr[0], kp_x, corner_tl[0], corner_br[1], kp_y, corner_tl[1])
                            # cv.circle(im_with_keypoints, (int(kp_x), int(kp_y)), 5, (255, 0, 0), -1)
                        if (abs(row - firstCorner[0]) <= 2) and (abs(col - firstCorner[1]) <= 2) and firstCornerFound:
                            print([row, col], " vs ", [firstCorner[0], firstCorner[1]])
                            color = (0, 255, 0)
                            # print("Also blocked")
                        else:
                            print([row, col], " vs ", [firstCorner[0], firstCorner[1]])
                            color = (0, 0, 255)
                            # print("Out of range")
                        cv.circle(im_with_keypoints, (int((corner_tl[0] + corner_tr[0])/2), int((corner_tl[1] + corner_br[1])/2)), 5, color, -1)
                    
                cv.imshow("Cup detection", im_with_keypoints)


        # # Check if the blob's center is between adjacent corners
        # for i in range(len(validCorners)-1):
        #     # Get two consecutive validCorners (you can adjust this for grid alignment)
        #     corner1 = validCorners[i][0]
        #     corner2 = validCorners[i+1][0]
            
        #     # Create a bounding box between the two validCorners (for simplicity, we use axis-aligned bounding box)
        #     min_x, max_x = min(corner1[0], corner2[0]), max(corner1[0], corner2[0])
        #     min_y, max_y = min(corner1[1], corner2[1]), max(corner1[1], corner2[1])
            
        #     boxFilled = False

        #     for keypoint in keypoints:
        #         kp_x, kp_y = keypoint.pt
        #         # Check if the keypoint's center lies within this bounding box
        #         if min_x <= kp_x <= max_x and min_y <= kp_y <= max_y:
        #             boxFilled = True
        #             # Draw a rectangle around the bounding box where the blob is located
        #             # cv.rectangle(im_with_keypoints, (int(min_x), int(min_y)), (int(max_x), int(max_y)), color = (0, 255, 0), thickness=2)
        #             cv.circle(im_with_keypoints, (int(kp_x), int(kp_y)), 5, (0, 0, 255), -1)  # Mark the blob center with a red circle
                    
        #     cv.rectangle(im_with_keypoints, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (255,0,0))
        #     cv.imshow("Cup detection", im_with_keypoints)
        


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