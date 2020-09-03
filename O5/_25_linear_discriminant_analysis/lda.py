import numpy as np

from O5 import prepare_data_wine

# 1. Standardize data

X_train, X_test, y_train, y_test = prepare_data_wine(standardize=True,
                                                     split=True)

# 2. Compute mean vector for each class

np.set_printoptions(precision=4)
mean_vectors = []
for label in np.sort(np.unique(y_train)):
    mean_vectors.append(np.mean(X_train[y_train == label]))
    print('MV %s: %s\n' % (label, mean_vectors[label - 1]))

