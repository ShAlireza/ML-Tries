import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from O5 import prepare_data_wine

# 1. Standardizing data

X_train, X_test, y_train, y_test, df_wine = prepare_data_wine(standardize=True,
                                                              split=True,
                                                              dataframe=True)

# 2. Constructing covariance matrix

# After completing the mandatory preprocessing by executing the preceding code,
# let's advance to the second step: constructing the covariance matrix.
# The symmetric d×d-dimensional covariance matrix, where d is the number of
# dimensions in the dataset, stores the pairwise covariances between the
# different features. For example, the covariance between two features, x(j)
# and x(k) , on the population level can be calculated via the following
# equation:
#       σ(jk) = (1 / (n - 1) ) * ∑(i=1, n) (x(j, i) - μ(j))(x(k, i) - μ(k))

# Here, μμ jj and μμ kk are the sample means of features j and k, respectively.
# Note that the sample means are zero if we standardized the dataset. A
# positive covariance between two features indicates that the features increase
# or decrease together, whereas a negative covariance indicates that the
# features vary in opposite directions. For example, the covariance matrix of
# three features can then be written as follows (note that Σ stands for the
# Greek uppercase letter sigma, which is not to be confused with summation
# symbol:
#       Σ = [σ(1)^2 σ(1,2) σ(1, 3);σ(2,1) σ(2)^2 σ(2, 3);σ(3,1) σ(3, 2) σ(3)^2]

# The eigenvectors of the covariance matrix represent the principal components
# (the directions of maximum variance), whereas the corresponding eigenvalues
# will define their magnitude. In the case of the Wine dataset, we would obtain
# 13 eigenvectors  and eigenvalues from the 13 × 13 -dimensional covariance
# matrix.

# Now, for our third step, let's obtain the eigenpairs of the covariance
# matrix. As you will remember from our introductory linear algebra classes, an
# eigenvector, v, satisfies the following condition:
#       Σv = λv
# Here, λ is a scalar: the eigenvalue.

cov_mat = np.cov(X_train.T)

# 3. Obtaining the eigenvalues and eigenvectors of the covariance matrix.

eigen_values, eigen_vectors = np.linalg.eig(cov_mat)
print(f"EigenValues \n{eigen_values}")

# The numpy.linalg.eig function was designed to operate on
# both symmetric and non-symmetric square matrices. However,
# you may find that it returns complex eigenvalues in certain
# cases.
# A related function, numpy.linalg.eigh , has been
# implemented to decompose Hermetian matrices, which is a
# numerically more stable approach to working with symmetric
# matrices such as the covariance matrix; numpy.linalg.eigh
# always returns real eigenvalues.


# Since we want to reduce the dimensionality of our dataset by compressing it
# onto a new feature subspace, we only select the subset of the eigenvectors
# (principal components) that contains most of the information (variance). The
# eigenvalues define the magnitude of the eigenvectors, so we have to sort the
# eigenvalues by decreasing magnitude; we are interested in the top k
# eigenvectors based on the values of their corresponding eigenvalues. But
# before we collect those k most informative eigenvectors, let's plot the
# variance explained ratios of the eigenvalues. The variance explained ratio of
# an eigenvalue, λ(j) , is simply the fraction of an eigenvalue, λ(j) and the
# total sum of eigenvalues:
#       Explained variance ratio = λ(j) / ∑(i=1, d)λ(i)

# Using the NumPy cumsum function, we can then calculate the cumulative sum of
# explained variances, which we will then plot via Matplotlib's step function:

total = sum(eigen_values)
explained_variance = [(i / total) for i in sorted(eigen_values, reverse=True)]
cumulative_sum_explained_variance = np.cumsum(explained_variance)

plt.bar(range(1, 14), explained_variance, alpha=0.5, align='center',
        label='Individual explained variance')

plt.step(range(1, 14), cumulative_sum_explained_variance, where='mid',
         label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# The resulting plot indicates that the first principal component alone
# accounts for approximately 40 percent of the variance.
# Also, we can see that the first two principal components combined explain
# almost 60 percent of the variance in the dataset
# we should remind ourselves that PCA is an unsupervised method, which means
# that information about the class labels is ignored. Whereas a random forest
# uses the class membership information to compute the node impurities,
# variance measures the spread of values along a feature axis.

# 4. Sorting the eigenvalues by decreasing order to rank the eigenvectors
eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i])
               for i in range(len(eigen_values))]

eigen_pairs = sorted(eigen_pairs, key=lambda x: x[0], reverse=True)

# 5. Select k(here k=2) eigenvectors, which correspond to the k largest
# eigenvalues, where k is the dimensionality of the new feature subspace
# (k <= d)

# Now, we collect the two eigenvectors that correspond to the two largest
# eigenvalues, to capture about 60 percent of the variance in this dataset.
# Note that two eigenvectors have been chosen for the purpose of illustration,
# since we are going to plot the data via a two-dimensional scatter plot later
# in this subsection. In practice, the number of principal components has to be
# determined by a tradeoff between computational efficiency and the performance
# of the classifier.
first_eigenvalue_vector = eigen_pairs[0][1]
second_eigenvalue_vector = eigen_pairs[1][1]

# 6. Construct a projection matrix, W, from the "top" k eigenvectors

w = np.hstack((first_eigenvalue_vector[:, np.newaxis],
               second_eigenvalue_vector[:, np.newaxis]))
print(w)

# 7. Transform the d-dimensional input dataset, X, using the projection matrix,
# W, to obtain the new k-dimensional feature subspace.
#       Transformation: x' = xW for x ∈ X
print(X_train[0].dot(w))

# Or full transformation: X' = XW

X_train_pcs = X_train.dot(w)
colors = ['red', 'blue', 'green']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pcs[y_train == l, 0],
                X_train_pcs[y_train == l, 1],
                c=c, label=l, marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
# As we can see in the resulting plot, the data is more spread along the x-axis
# —the first principal component—than the second principal component (y-axis),
# which is consistent with the explained variance ratio plot that we created in
# the previous subsection. However, we can tell that a linear classifier will
# likely be able to separate the classes well.
# Although we encoded the class label information for the purpose of
# illustration in the preceding scatter plot, we have to keep in mind that PCA
# is an unsupervised technique that doesn't use any class label information.
