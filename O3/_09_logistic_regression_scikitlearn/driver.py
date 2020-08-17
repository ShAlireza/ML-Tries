import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from utils import plot_decision_regions

from O3 import prepare_data

X, Y = prepare_data()

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=1, stratify=Y
)

# Standardize data

sc = StandardScaler()
sc.fit(X_train)

X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

# C stands for regularization constant, smaller values specify stronger
# regularization.

# solver: Chosen optimization algorithm from varieties of algorithms
# provided by sklearn: newton-cg, lbfgs, liblinear, sag, saga

logistic_regression = LogisticRegression(C=100.0, random_state=1,
                                         solver='lbfgs', multi_class='ovr')

logistic_regression.fit(X_train, Y_train)

# Combine train and test dataset for plotting together

X_combined = np.vstack((X_train, X_test))
Y_combined = np.hstack((Y_train, Y_test))

plot_decision_regions(X=X_combined, Y=Y_combined,
                      classifier=logistic_regression,
                      test_idx=range(105, 150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

print(logistic_regression.predict_proba(X_test[:3, :]))

# print(logistic_regression.predict_proba(X_test[:3, :]).argmax(axis=1))

print(logistic_regression.predict(X_test[:3, :]))

# reshape(1, -1) to make one dimensional array to a two dimensional array
print(logistic_regression.predict(X_test[0, :].reshape(1, -1)))
