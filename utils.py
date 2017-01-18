import numpy as np
import pandas as pd
import cv2

def reshape_data(label, features_per_window):
    # Converts (n, 1)-shaped table in csv into (m, f) shape
    # where m is number of observations, f is number of features.
    # Input: class label (int), number of frames in sliding window (int) 

    # Reshape is easier in numpy
    data = np.loadtxt('./data/train.csv')

    # Take contents of each sliding window as an observation
    num_rows = int(len(data)/features_per_window) # int() rounds down
    data = data[:num_rows*features_per_window] # drop final window to prevent truncation
    data = np.reshape(data, (num_rows, features_per_window))

    # Place labels as the final column of the data
    label_vec = np.asarray([label]*num_rows)
    data = np.c_[data, label_vec]

    # Read/write csv is easier pandas
    df = pd.DataFrame(data)
    df.to_csv('./data/train' + str(label) + '.csv', index=False)

    print "\nPreview of the most recent training data:"
    print df

def combine_data(num_classes):
    # Combines training data for separate classes into a single dataset.
    # Input: number of classes in training data
    # Output: data, labels

    data = pd.read_csv('./data/train0.csv')

    for label in range(num_classes - 1):
        data = data.append(pd.read_csv('./data/train' + str(label + 1) + '.csv'), ignore_index=True)

    return data.iloc[:, :-1], data.iloc[:, -1]

def print_bbox(img, k, d):
    print("Face {}: Left: {} Top: {} Right: {} Bottom: {}".format(
          k+1, d.left(), d.top(), d.right(), d.bottom()))

def print_parts(shape):
    print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                              shape.part(1)))

def plot_bbox(img, left, right, top, bottom, color=(0, 255, 255)):
    cv2.circle(img, (left, int(top + (bottom - top)/2.0)), 5, color)
    cv2.circle(img, (right, int(top + (bottom - top)/2.0)), 5, color)
    cv2.circle(img, (int(left + (right - left)/2.0), top), 5, color)
    cv2.circle(img, (int(left + (right - left)/2.0), bottom), 5, color)
                    
def plot_landmarks(img, parts, black_bg=False, color=(0, 255, 255), resolution=(480, 640)):
    if black_bg == True:
        row = [[0, 0, 0]]*resolution[1]
        img = np.asarray([row]*resolution[0])
        img = img.astype(np.uint8) # need uint8 for cv2.circle

    for i in range(len(parts)):
        cv2.circle(img, tuple(parts[i]), 1, color)

    cv2.imshow("preview", img)

if __name__ == "__main__":
    reshape_data(0)
