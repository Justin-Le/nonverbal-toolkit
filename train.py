import sys
import time
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
        classifier = 'lr'
        param1 = 'l2'
        param2 = 1.0
        seed = 0
    else:
        num_classes = int(sys.argv[1])
        classifier = str(sys.argv[2])
        param1 = sys.argv[3]
        param2 = float(sys.argv[4])
        seed = int(sys.argv[5])
     
    X_train, Y_train = combine_data(num_classes)

    if classifier == 'lr':
        lr = LogisticRegression(penalty=str(param1), C=param2, random_state=seed)
        start = time.time()
        score = cross_val_score(lr, X_train, Y_train, cv=10, scoring="accuracy")
        cv_time = time.time() - start

        print score
        print np.mean(score)
        print np.var(score)
        print cv_time

        lr.fit(X_train.iloc[:, 1:], Y_train)
        joblib.dump(lr, './models/logistic_regression.pkl')

    elif classifier == 'rf':
        rf = RandomForestClassifier(n_estimators=int(param1), max_features=param2, random_state=seed)
        start = time.time()
        score = cross_val_score(rf, X_train, Y_train, cv=10, scoring="accuracy")
        cv_time = time.time() - start

        print score
        print np.mean(score)
        print np.var(score)
        print cv_time

        rf.fit(X_train.iloc[:, 1:], Y_train)
        joblib.dump(rf, './models/random_forest.pkl')

if __name__ == "__main__":
    train()
