import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from utils import plot_decision_regions

from O3 import prepare_data, LogisticRegressionGD

X, Y = prepare_data()

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=1, stratify=Y
)

# Standardized features data

sc = StandardScaler()
sc.fit(X_train)

X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

# Extract only class 0 and class 1 Iris flowers

X_train_01_subset = X_train[(Y_train == 0) | (Y_train == 1)]
Y_train_01_subset = Y_train[(Y_train == 0) | (Y_train == 1)]

X_test_01_subset = X_test[(Y_test == 0) | (Y_test == 1)]
Y_test_01_subset = Y_test[(Y_test == 0) | (Y_test == 1)]

logistic_regression = LogisticRegressionGD(eta=0.05, epochs=1000,
                                           random_state=1)

logistic_regression.fit(X_train_01_subset, Y_train_01_subset)

plot_decision_regions(X=X_train_01_subset, Y=Y_train_01_subset,
                      classifier=logistic_regression)

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')

plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

accuracy = accuracy_score(
    Y_test_01_subset,
    logistic_regression.predict(X_test_01_subset)
)
print(f'Accuracy Score is: {accuracy * 100}')
