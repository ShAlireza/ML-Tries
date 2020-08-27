"""

    L2 regularization is one approach to reduce the complexity of a model by
    penalizing large individual weights. We defined the squared L2 norm of our
    weight vector, w, as follows:
        L2: ‖ww‖ = ∑ w(j)^2

    L2 regularization adds a penalty term to the cost function that effectively
    results in less extreme weight values compared to a model trained with an
    unregularized cost function.

    Another approach to reduce the model complexity is the related L1
    regularization:
        L1: ‖ww‖ = ∑|w(j)|

    Here, we simply replaced the square of the weights by the sum of the
    absolute values of the weights. In contrast to L2 regularization, L1
    regularization usually yields sparse feature vectors and most feature
    weights will be zero. Sparsity can be useful in practice if we have a
    high-dimensional dataset with many features that are irrelevant, especially
    in cases where we have more irrelevant dimensions than training examples.
    In this sense, L1 regularization can be understood as a technique
    for feature selection.

"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from O4 import prepare_data

X_train, X_test, y_train, y_test, df_wine = prepare_data(standardize=True,
                                                         split=True,
                                                         dataframe=True)

# For regularized models in scikit-learn that support L1 regularization, we
# can simply set the penalty parameter to 'l1' to obtain a sparse solution
# Note that we also need to select a different optimization algorithm
# (for example, solver='liblinear' ), since 'lbfgs' currently does not
# support L1-regularized loss optimization.

logistic_regression = LogisticRegression(penalty='l1', C=1.0,
                                         solver='liblinear', multi_class='ovr')
logistic_regression.fit(X_train, y_train)

print(f'Training accuracy: {logistic_regression.score(X_train, y_train)}')

print(f'Test accuracy: {logistic_regression.score(X_test, y_test)}')

# When we access the intercept terms via the lr.intercept_ attribute
# Since we fit the LogisticRegression object on a multi class dataset via the
# one-vs.-rest (OvR) approach, the first intercept belongs to the model that
# fits class 1 versus classes 2 and 3, the second value is the intercept of
# the model that fits class 2 versus classes 1 and 3, and the third value is
# the intercept of the model that fits class 3 versus classes 1 and 2

print(logistic_regression.intercept_)

# The weight array that can be accessed via the lr.coef_ attribute contains
# three rows of weight coefficients, one weight vector for each class. Each
# row consists of 13 weights, where each weight is multiplied by the respective
# feature in the 13-dimensional Wine dataset to calculate the net input

print(logistic_regression.coef_)

# In scikit-learn, the intercept_ corresponds to ww 0 and coef_ corresponds to
# the values w(j) for j > 0.


fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan',
          'magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'lightblue',
          'gray', 'indigo', 'orange']

weights, params = [], []

for c in np.arange(-4., 6.):
    lr = LogisticRegression(penalty='l1', C=10.0 ** c, solver='liblinear',
                            multi_class='ovr', random_state=0)
    lr.fit(X_train, y_train)
    weights.append(lr.coef_[1])
    params.append(10 ** c)

weights = np.array(weights)

for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column], label=df_wine.columns[column + 1],
             color=color)

plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10 ** -5, 10 ** 5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03),
          ncol=1, fancybox=True)

plt.show()
