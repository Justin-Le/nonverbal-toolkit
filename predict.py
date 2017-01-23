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
import cv2
from sklearn.externals import joblib

from extract_features import extract_features
from utils import reshape_data, print_bbox, print_parts, plot_bbox, plot_landmarks

def predict():
    if len(sys.argv) < 2:
        classifier = 'lr'
    elif len(sys.argv) == 2:
        classifier = str(sys.argv[1])
    else:
        print(
            "\nUsage:\n"
            "\npython predict.py [classifier]\n\n"

            "\nChoose `classifier` as `lr` for logistic regression or `rf` for random forest, depending on the one chosen when executing `train.py` previously.\n")
        exit()

    # Load trained facial model
    detector = dlib.get_frontal_face_detector()
    predictor_path = "models/face_predictor.dat"
    predictor = dlib.shape_predictor(predictor_path)

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

    # Initialize data in first window
    X_test = np.array([])

    # Load trained model
    if classifier == 'lr':
        clf = joblib.load('./models/logistic_regression.pkl')
    elif classifier == 'rf':
        clf = joblib.load('./models/random_forest.pkl')

    # Initialize trajectories of top/left points of the facial position-in-frame
    top_trajectory = np.array([])
    left_trajectory = np.array([])

    # Perform predictions on video
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
                    ######################################## 
                    # FEATURE EXTRACTION
                    ######################################## 

                    # Extract facial position-in-frame
                    top = d.top()
                    left = d.left()

                    # Create feature vector
                    # Continue stacking features here as needed
                    features = extract_features(predictor, img, d)
                    top_trajectory = np.hstack((top_trajectory, top))
                    left_trajectory = np.hstack((left_trajectory, left))
                    X_test = np.hstack((X_test, features))
     
                    if frame_count >= time_window:
                        # Append the variance of position-in-frame to the end of each window
                        # X_test is the feature vector for the entire window
                        #  X_test = np.hstack((X_test, np.var(top_trajectory)))
                        #  X_test = np.hstack((X_test, np.var(left_trajectory)))
                        #  print X_test[-2]
                        #  print X_test[-1]

                        # Reshape needed: 1d test data is deprecated
                        X_test = X_test[:136*time_window].reshape(1, -1)
                        y_pred = clf.predict(X_test)

                        if (y_pred == 0):
                            print "\nNeutral.\n"
                        elif (y_pred == 1):
                            print "\nYay!\n"
                        elif (y_pred == 2):
                            print "\nWow!\n"
                        elif (y_pred == 3):
                            print "\nI agree.\n"
                        elif (y_pred == 4):
                            print "\nHa! Ha! Ha!\n"
                        elif (y_pred == 5):
                            print "\nI disagree.\n"
                        elif (y_pred == 6):
                            print "\nStop!\n"

                        # Reshape into 1d again,
                        # discard oldest vector from window,
                        # and discard the two position variances from window
                        X_test = X_test[0, 136:]
                        
                        # Discard oldest positions from trajectories
                        top_trajectory = top_trajectory[1:]
                        left_trajectory = left_trajectory[1:]
     
                    cv2.imshow("preview", img)
                    
                    key = cv2.waitKey(1)

            if time.time() - start >= 1:
                start = time.time()

    # Stop predicting when user inputs Ctrl-c
    except KeyboardInterrupt:
        pass

    print "Done."

if __name__ == "__main__":
    predict()
