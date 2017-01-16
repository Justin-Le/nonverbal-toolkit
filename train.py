import sys
import numpy as np
import pandas as pd

if len(sys.argv) < 2:
    num_classes = 2
elif len(sys.argv) == 2:
    num_classes = int(sys.argv[1])
else:
    print(
        "\nUsage:\n"
        "python train.py [number of classes]\n")
    exit()
 
data = pd.read_csv('./data/train0')

for label in range(num_classes - 1):
    data = data.append(pd.read_csv('./data/train' + str(label + 1)), ignore_index=True)

print data
