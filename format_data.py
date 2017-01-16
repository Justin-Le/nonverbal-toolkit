import numpy as np
import pandas as pd

def format_data(label, time_window):
    data = np.loadtxt('./data/train.csv')

    num_frames = len(data)/136

    # If video clip too short to fill the sliding window,
    # take entire clip as an observation
    if time_window > num_frames:
        time_window = num_frames

    # Take contents of each sliding window as an observation
    num_features = 136*time_window
    num_rows = int(len(data)/(num_features)) # int() rounds down
    data = data[:num_rows*num_features] # prevent window from being truncated
    data = np.reshape(data, (num_rows, num_features))

    # Place labels as the final column of the data
    label_vec = np.asarray([label]*num_rows)
    data = np.c_[data, label_vec]

    df = pd.DataFrame(data)
    df.to_csv('./data/train' + str(label))
    # np.savetxt('./data/train' + str(label) + '.csv', data, delimiter=",")

    print "\nPreview of the most recent training data:"
    print df

if __name__ == "__main__":
    format_data(0)
