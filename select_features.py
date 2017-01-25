import numpy as np

def select_features(X, time_window):
    X = np.asarray(X)

    X_select = X[:, [0, 16, 20, 23, 38, 43]]

    for i in range(time_window - 1):
        frame = X[:, [i + 1, i + 17, i + 21, i + 24, i + 39, i + 44]]
        X_select = np.hstack((X_select, frame))

    return X_select
