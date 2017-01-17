import sys
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris
from sklearn.externals import joblib

from utils import combine_data

def train():
    if len(sys.argv) < 2:
        num_classes = 2
    elif len(sys.argv) == 2:
        num_classes = int(sys.argv[1])
    else:
        print(
            "\nUsage:\n"
            "python train.py [number of classes]\n")
        exit()
     
    X_train, Y_train = combine_data(num_classes)

    lr = LogisticRegression()
    print cross_val_score(lr, X_train, Y_train, cv=5, scoring="accuracy")

    lr.fit(X_train.iloc[:, 1:], Y_train)
    joblib.dump(lr, './models/logistic_regression.pkl')

if __name__ == "__main__":
    train()
