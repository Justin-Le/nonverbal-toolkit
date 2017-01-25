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

from extract_features import extract_features
from utils import reshape_data, print_bbox, print_parts, plot_bbox, plot_landmarks

def sample():
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

    # Load trained facial model
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

    # Initialize trajectories of top/left points of the facial position-in-frame
    top_trajectory = np.array([])
    left_trajectory = np.array([])

    # Record data for the desired class
    try:
        while rval:
            frame_count += 1

            rval, img = vc.read()
            img = np.asarray(img)

            # Ask the detector to find the bounding boxes of each face.
            # The argument of 1 indicates that we should upsample the image 1 time.
            # A higher number allows for the detection of more faces.
            # This step accounts for the majority of computation time.
            dets = detector(img, 0)

            # print("Number of faces detected: {}".format(len(dets)))

            if len(dets) > 0:
                for k, d in enumerate(dets):
                    # print_bbox(img, k, d)

                    features, parts = extract_features(predictor, img, d)
                    keypoints = parts[0] # left edge
                    keypoints = np.vstack((keypoints, parts[20])) # right edge
                    keypoints = np.vstack((keypoints, parts[16])) # left brow
                    keypoints = np.vstack((keypoints, parts[23])) # right brow
                    keypoints = np.vstack((keypoints, parts[38])) # left eye
                    keypoints = np.vstack((keypoints, parts[43])) # right eye
                    features = keypoints.reshape(1, 12)[0]
                    print features
                    
                    """
                    top_trajectory = np.hstack((top_trajectory, top))
                    left_trajectory = np.hstack((left_trajectory, left))

                    if frame_count >= time_window:
                        # Append the variance of position-in-frame to the end of each window
                        features = np.hstack((features, np.var(top_trajectory)))
                        features = np.hstack((features, np.var(left_trajectory)))

                        # Reset trajectories and frame counter
                        top_trajectory = np.array([])
                        left_trajectory = np.array([])
                        frame_count = 0
                    """
 
                    # Append feature vector to csv
                    pd.DataFrame(features).to_csv('./data/train.csv', mode='a', header=False, index=False)

                    # Plot left, right, top, bottom coordinates of detected face
                    plot_bbox(img, d.left(), d.right(), d.top(), d.bottom(), color=(0, 255, 255))

                    plot_landmarks(img, keypoints, black_bg=False, color=(0, 255, 255), resolution=(480, 640))
                    
                    key = cv2.waitKey(1)

            if time.time() - start >= 1:
                print frame_count
                start = time.time()

    # Stop recording when user inputs Ctrl-c
    except KeyboardInterrupt:
        pass
        
    # Add 2 to account for the variance of position-in-frame
    # features_per_window = 136*time_window + 2

    features_per_window = 12*time_window

    reshape_data(label, features_per_window)
    os.system("rm ./data/train.csv")

    print "Done."

if __name__ == "__main__":
    sample()
