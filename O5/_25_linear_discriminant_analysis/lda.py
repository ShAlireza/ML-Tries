import numpy as np
import matplotlib.pyplot as plt

from O5 import prepare_data_wine

# 1. Standardize data

X_train, X_test, y_train, y_test = prepare_data_wine(standardize=True,
                                                     split=True)

# 2. Compute mean vector for each class

np.set_printoptions(precision=4)
mean_vectors = []
for label in np.sort(np.unique(y_train)):
    mean_vectors.append(np.mean(X_train[y_train == label], axis=0))
    print('MV %s: %s\n' % (label, mean_vectors[label - 1]))

# 3.1 Create within-class scatter matrix

d = 13  # number of features
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vectors):
    class_scatter = np.zeros((d, d))

    for row in X_train[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)
        class_scatter += (row - mv).dot((row - mv).T)
        S_W += class_scatter

print(f"Within-class scatter matrix: {S_W.shape[0]} x {S_W.shape[1]}")

# The assumption that we are making when we are computing the scatter matrices
# is that the class labels in the training dataset are uniformly distributed.
# However, if we print the number of class labels, we see that this assumption
# is violated:
print(f'Class label distribution: {np.bincount(y_train)[1:]}')

# Thus, we want to scale the individual scatter matrices, S(i), before we sum
# them up as scatter matrix S(W). When we divide the scatter matrices by the
# number of class-examples, nn ii , we can see that computing the scatter
# matrix is in fact the same as
# computing the covariance matrix, Σ(i) —the covariance matrix is a normalized
# version of the scatter matrix:
# Σ(i) = 1/n(i) * S(i) =1/n(i) * ∑(x−m(i))(x-m(i))^T

# Compute scaled within-class scatter matrix:

d = 13
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vectors):
    class_scatter = np.cov(X_train[y_train == label].T)
    S_W += class_scatter
print(f"Within-class scatter matrix: {S_W.shape[0]} x {S_W.shape[1]}")

# 3.2 Compute between-class scatter matrix:

mean_overall = np.mean(X_train, axis=0)
d = 13  # Number of features
S_B = np.zeros((d, d))

for i, mean_vector in enumerate(mean_vectors):
    n = X_train[y_train == i + 1, :].shape[0]
    mean_vector = mean_vector.reshape(d, 1)  # Make column vector
    mean_overall = mean_overall.reshape(d, 1)
    S_B += n * (mean_vector - mean_overall).dot((mean_vector - mean_overall).T)

print(f'Between-class scatter matrix: {S_B.shape[0]}x{S_B.shape[1]}')

# 4 Compute the eigenvectors and corresponding eigenvalues of the matrx,
# S(w, -1) * S(b)
eigen_values, eigen_vectors = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

# 5 Sort the eigenvalues in descending order
eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i])
               for i in range(len(eigen_values))]
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
print(f'Eigenvalues in descending order: \n')
for eigen_value in eigen_pairs:
    print(eigen_value[0])

# In LDA, the number of linear discriminants is at most c−1, where c is the
# number of class labels, since the in-between scatter matrix, S(B), is the
# sum of c matrices with rank one or less. We can indeed see that we only have
# two nonzero eigenvalues (the eigenvalues 3-13 are not exactly zero, but this
# is due to the floating-point arithmetic in NumPy).

tot = sum(eigen_values.real)
discr = [(i / tot) for i in sorted(eigen_values.real, reverse=True)]
cum_discr = np.cumsum(discr)

plt.bar(range(1, 14), discr, alpha=0.5, align='center',
        label='Individual "discriminability"')
plt.step(range(1, 14), cum_discr, where='mid',
         label='Cumulative "discriminablity"')
plt.ylabel('"Discriminability" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# 6 Choose the k eigenvectors that correspond to the k largest eigenvalues
# to construct a d * k - dimensional transformation matrix, W; the
# eigenvectors are the columns of this matrix.

w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
               eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W: \n', w)

# 7 Project the examples onto the new feature space using the transformation
# matrix, W.

X_train_lda = X_train.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train == l, 0],
                X_train_lda[y_train == l, 1] * (-1),
                c=c, label=l, marker=m)

plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
