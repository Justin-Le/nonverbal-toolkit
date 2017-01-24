import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.decomposition import PCA

fig, axarr = plt.subplots(8, sharex=True)


for s in range(7):
    X = pd.read_csv('./data/train' + str(s) + '.csv')
    X = np.asarray(X)
    X = X[:, -2:]

    """
    pca = PCA(n_components=10, random_state=0)
    pca.fit(X)
    X = pca.transform(X)
    """

    step = range(X.shape[0])

    for i in range(X.shape[1]):
        axarr[s].plot(step, X[:, i])

plt.tight_layout()
plt.show()
