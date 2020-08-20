import numpy as np

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from utils import plot_decision_regions

from O3 import prepare_data

X_train, X_test, y_train, y_test = prepare_data(standardize=True,
                                                split=True)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

# n_jobs => number of cores using for parallel computing

forest = RandomForestClassifier(criterion='gini', n_estimators=25,
                                random_state=1, n_jobs=2)

forest.fit(X_train, y_train)
plot_decision_regions(X_combined, y_combined, classifier=forest,
                      test_idx=range(105, 150))

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')

plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
