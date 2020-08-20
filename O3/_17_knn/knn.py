import numpy as np

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from utils import plot_decision_regions

from O3 import prepare_data

X_train, X_test, y_train, y_test = prepare_data(standardize=True,
                                                split=True)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')

knn.fit(X_train, y_train)

plot_decision_regions(X_combined, y_combined, classifier=knn,
                      test_idx=range(105, 150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
