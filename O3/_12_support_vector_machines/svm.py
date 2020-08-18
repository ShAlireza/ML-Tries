import numpy as np

import matplotlib.pyplot as plt

from O3 import prepare_data

from utils import plot_decision_regions

from sklearn.svm import SVC

X_train, X_test, Y_train, Y_test = prepare_data(standardize=True,
                                                split=True)

svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train, Y_train)

X_combined = np.vstack((X_train, X_test))
Y_combined = np.hstack((Y_train, Y_test))

plot_decision_regions(X_combined, Y_combined, classifier=svm,
                      test_idx=range(105, 150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')

plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

print(f'Accuracy: {svm.score(X_test, Y_test) * 100}')
