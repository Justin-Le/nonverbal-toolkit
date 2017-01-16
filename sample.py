#!/usr/bin/python
#
#   Real-time facial feature detection with 68 landmarks, including
#   corners of the mouth, eyebrows, nose, and eyes.
#
#   Histogram of Oriented Gradients (HOG) features, linear classifier, image pyramid,
#   and sliding windows.  
#
#   Pose estimator taken from dlib's implementation of
#   One Millisecond Face Alignment with an Ensemble of Regression Trees,
#   Kazemi and Sullivan, CVPR 2014,
#   trained on the iBUG 300-W facial landmark dataset.
#
#   Train your own models using dlib's tools (e.g., train_shape_predictor.py).
#
#   Trained model can be obtained from:
#   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

import sys
import os
import dlib
import glob
import time

import numpy as np
import pandas as pd
import cv2

from utils import reshape_data

if len(sys.argv) < 2:
    predictor_path = "models/face_predictor.dat"
elif len(sys.argv) == 2:
    predictor_path = sys.argv[1]
else:
    print(
        "\nUsage:\n"
        "python sample.py models/face_predictor.dat\n"
        "\nExample face predictor:\n"
        "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n")
    exit()

# Load trained model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Request user input for the label of the desired class in the training data
try:
    label = int(raw_input('Enter the number corresponding to the label of the action you would like to record.\n'))
except ValueError:
    print "\nPlease enter an integer.\n"

# Request user input for the size of the sliding window
try:
    time_window = int(raw_input('Enter the duration (in number of frames) of the action. This must match the duration of other actions in the data.\n'))
except ValueError:
    print "\nPlease enter an integer greater than 0.\n"

# Initialize video input
vc = cv2.VideoCapture(0)
cv2.namedWindow("preview")

if vc.isOpened():
    rval, img = vc.read()
else:
    rval = False

frame_count = 0
start = time.time()

# Record data for the desired class
try:
    while rval:
        frame_count += 1

        rval, img = vc.read()
        img = np.asarray(img)

        # Ask the detector to find the bounding boxes of each face. 
        # The argument of 1 indicates that we should upsample the image 1 time, 
        # allowing for the detection of more faces.
        # This step accounts for the majority of computation time.
        dets = detector(img, 0)

        # print("Number of faces detected: {}".format(len(dets)))

        if len(dets) > 0:
            for k, d in enumerate(dets):
                print("Face {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                    k+1, d.left(), d.top(), d.right(), d.bottom()))

                ######################################## 
                # FEATURE EXTRACTION
                ######################################## 

                shape = predictor(img, d)

                print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                          shape.part(1)))

                # Extract (x, y) coordinates of facial landmarks
                parts = [[shape.part(n).x, shape.part(n).y] for n in range(shape.num_parts)]
                parts = np.asarray(parts).astype(int)
                
                # Extract facial position-in-frame
                position = d.top() # simple

                # Compute landmark coordinates with respect to position-in-frame
                # to enforce translation invariance of features (roughly, due to noise)
                parts_x = parts.T[0] - d.left()
                parts_y = parts.T[1] - d.top()

                # Create feature vector
                # Continue stacking features here as needed
                features = np.hstack((parts_x, parts_y))
                features = np.hstack((features, position))
                num_features = len(features)
 
                # Append feature vector to csv
                pd.DataFrame(features).to_csv('./data/train.csv', mode='a', header=False, index=False)

                ######################################## 
                # PLOTTING
                ######################################## 

                # Uncomment these 3 lines to plot landmarks over a black background
                row = [[0, 0, 0]]*640
                img = np.asarray([row]*480)
                img = img.astype(np.uint8) # need uint8 for cv2.circle

                # Plot landmarks
                for i in range(len(parts)):
                    cv2.circle(img, tuple(parts[i]), 1, (0, 255, 255))

                # Plot left, right, top, bottom coordinates of detected face
                cv2.circle(img, (d.left(), int(d.top() + (d.bottom() - d.top())/2.0)), 5, (0, 255, 255))
                cv2.circle(img, (d.right(), int(d.top() + (d.bottom() - d.top())/2.0)), 5, (0, 255, 255))
                cv2.circle(img, (int(d.left() + (d.right() - d.left())/2.0), d.top()), 5, (0, 255, 255))
                cv2.circle(img, (int(d.left() + (d.right() - d.left())/2.0), d.bottom()), 5, (0, 255, 255))
                
                cv2.imshow("preview", img)
                
                key = cv2.waitKey(1)

        if time.time() - start >= 1:
            print frame_count
            start = time.time()

# Stop recording when user inputs Ctrl-c
except KeyboardInterrupt:
    pass
    
reshape_data(label, num_features, time_window)
os.system("rm ./data/train.csv")

print "Done."
