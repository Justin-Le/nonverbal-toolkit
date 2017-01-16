import sys
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from utils import combine_data

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

print X_train
print Y_train

lr = LogisticRegression()
lr.fit(X_train, Y_train)
print lr.get_params()

Y_pred = lr.predict(X_train)
print classification_report(Y_pred, Y_train)
