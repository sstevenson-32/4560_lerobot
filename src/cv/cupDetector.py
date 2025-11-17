import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def getCupPos():
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
            cornersFound = [[],[]]

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
                            
                            # if (-1 <= (row - firstCorner[0]) <= 2) and (-1 <= (col - firstCorner[1]) <= 2) and firstCornerFound:    
                            if (abs(row - firstCorner[0]) + abs(col - firstCorner[1]) <= 2) and firstCornerFound:
                                color = (0, 255, 0)
                                # cornersFound[0].append(int((corner_tl[0] + corner_tr[0])/2))
                                # cornersFound[1].append(int((corner_tl[1] + corner_br[1])/2))
                                cornersFound[0].append(row)
                                cornersFound[1].append(col)

                            else:
                                color = (0, 0, 255)
                            cv.circle(im_with_keypoints, (int((corner_tl[0] + corner_tr[0])/2), int((corner_tl[1] + corner_br[1])/2)), 5, color, -1)

                        # cv.circle(im_with_keypoints, (int((corner_tl[0] + corner_tr[0])/2), int((corner_tl[1] + corner_br[1])/2)), 5, color, -1)
                        cv.imshow("Cup detection", im_with_keypoints)
            center = (int(np.mean(cornersFound[0])), int(np.mean(cornersFound[1])))
            center = (np.mean(cornersFound[0]), np.mean(cornersFound[1]))
            cv.circle(im_with_keypoints, center, 5, (255, 0, 0), -1)
            cv.imshow("Cup detection", im_with_keypoints)
        cv.waitKey(50)
    
    cv.destroyWindow("CupDetection")
    vc.release()
    return center
