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
