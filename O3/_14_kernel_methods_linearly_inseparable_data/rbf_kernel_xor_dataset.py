import numpy as np

import matplotlib.pyplot as plt

from sklearn.svm import SVC

from utils import plot_decision_regions

from O3 import xor_dataset

"""
    One of the most widely used kernels is the radial basis function (RBF)
    kernel, which can simply be called the Gaussian kernel.

"""

X_xor, y_xor = xor_dataset()

svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)

svm.fit(X_xor, y_xor)

plot_decision_regions(X_xor, y_xor, classifier=svm)

plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

print(f"Accuracy: {svm.score(X_xor, y_xor) * 100}%")
