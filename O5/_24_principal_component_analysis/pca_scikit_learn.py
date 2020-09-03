import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from utils import plot_decision_regions

from O5 import prepare_data_wine

X_train, X_test, y_train, y_test = prepare_data_wine(standardize=True,
                                                     split=True)

# initializing the PCA transformer and
# logistic regression estimator:

pca = PCA(n_components=2)
logistic_regression = LogisticRegression(multi_class='ovr', random_state=1,
                                         solver='lbfgs')

# Dimensionality reduction:

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Fitting the logistic regression model on the reduced dataset:

logistic_regression.fit(X_train_pca, y_train)

plot_decision_regions(X_train_pca, y_train, classifier=logistic_regression)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

plot_decision_regions(X_test_pca, y_test, classifier=logistic_regression)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

# If we are interested in the explained variance ratios of the different
# principal components, we can simply initialize the PCA class with the
# n_components parameter set to None , so all principal components are kept and
# the explained variance ratio can then be accessed via the
# explained_variance_ratio_ attribute

pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train)
print(pca.explained_variance_ratio_)

# Note that we set n_components=None when we initialized the PCA class so that
# it will return all principal components in a sorted order, instead of
# performing a dimensionality reduction.


