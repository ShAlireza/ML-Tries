"""
Supervised data compression via linear discriminant analysis

    LDA can be used as a technique for feature extraction to increase the
    computational efficiency and reduce the degree of overfitting due to the
    curse of dimensionality in non-regularized models. The general concept
    behind LDA is very similar to PCA, but whereas PCA attempts to find the
    orthogonal component axes of maximum variance in a dataset, the goal in LDA
    is to find the feature subspace that optimizes class separability.

    Both PCA and LDA are linear transformation techniques that can be used to
    reduce the number of dimensions in a dataset; the former is an unsupervised
    algorithm, whereas the latter is supervised. Thus, we might think that LDA
    is a superior feature extraction technique for classification tasks
    compared to PCA. However, A.M. Martinez reported that preprocessing via PCA
    tends to result in better classification results in an image recognition
    task in certain cases, for instance, if each class consists of only a small
    number of examples (PCA Versus LDA, A. M. Martinez and A. C. Kak, IEEE
    Transactions on Pattern Analysis and Machine Intelligence, 23(2): 228-233,
    2001)

    One assumption in LDA is that the data is normally distributed. Also, we
    assume that the classes have identical covariance matrices and that the
    training examples are statistically independent of each other. However,
    even if one, or more, of those assumptions is (slightly) violated, LDA for
    dimensionality reduction can still work reasonably well (Pattern
    Classification 2nd Edition, R. O. Duda, P. E. Hart, and D. G.
    Stork, New York, 2001).

The inner workings of linear discriminant analysis

    1. Standardize the d-dimensional dataset (d is the number of features).
    2. For each class, compute the d-dimensional mean vector.
    3. Construct the between-class scatter matrix, S(B) , and the within-class
       scatter matrix, S(W)
    4. Compute the eigenvectors and corresponding eigenvalues of the matrix,
       S(W, -1) * S(B).
    5. Sort the eigenvalues by decreasing order to rank the corresponding
       eigenvectors.
    6. Choose the k eigenvectors that correspond to the k largest eigenvalues
       to construct a dd × kk -dimensional transformation matrix, W; the
       eigenvectors are the columns of this matrix.
    7. Project the examples onto the new feature subspace using the
       transformation matrix, W.

"""
