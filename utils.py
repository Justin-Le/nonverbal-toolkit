import numpy as np
import pandas as pd

def reshape_data(label, num_features, time_window):
    # Converts (n, 1)-shaped table in csv into (m, f) shape
    # where m is number of observations, f is number of features.
    # Input: class label (int), number of frames in sliding window (int) 

    # Reshape is easier in numpy
    data = np.loadtxt('./data/train.csv')

    num_frames = len(data)/num_features

    # If video clip too short to fill the sliding window,
    # take entire clip as an observation
    if time_window > num_frames:
        time_window = num_frames

    # Take contents of each sliding window as an observation
    num_features = num_features*time_window
    num_rows = int(len(data)/(num_features)) # int() rounds down
    data = data[:num_rows*num_features] # prevent window from being truncated
    data = np.reshape(data, (num_rows, num_features))

    # Place labels as the final column of the data
    label_vec = np.asarray([label]*num_rows)
    data = np.c_[data, label_vec]

    # Read/write csv is easier pandas
    df = pd.DataFrame(data)
    df.to_csv('./data/train' + str(label) + '.csv')

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

if __name__ == "__main__":
    reshape_data(0)
