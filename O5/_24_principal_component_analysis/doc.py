"""

    In this section, we will discuss PCA, an unsupervised linear transformation
    technique that is widely used across different fields, most prominently for
    feature extraction and dimensionality reduction. Other popular applications
    of PCA include exploratory data analyses and the denoising of signals in
    stock market trading, and the analysis of genome data and gene expression
    levels in the field of bioinformatics.

    PCA helps us to identify patterns in data based on the correlation between
    features. In a nutshell, PCA aims to find the directions of maximum
    variance in high-dimensional data and projects the data onto a new subspace
    with equal or fewer dimensions than the original one. The orthogonal axes
    (principal components) of the new subspace can be interpreted as the
    directions of maximum variance given the constraint that the new feature
    axes are orthogonal to each other.

    If we use PCA for dimensionality reduction, we construct a d×k-dimensional
    transformation matrix, W, that allows us to map a vector, x, the features
    of a training example, onto a new k-dimensional feature subspace that has
    fewer dimensions than the original d-dimensional feature space. For
    instance, the process is as follows. Suppose we have a feature vector, x:
        x = [x(1) , x(2) , ... , x(d)], x ∈ R(, d)

    which is then transformed by a transformation matrix, W ∈ R(, d×k):
        xW = z

    resulting the output vector:
        z = [z(1) , z(2) , ... , z(k)], z ∈ R(, k)


    As a result of transforming the original d-dimensional data onto this new
    k-dimensional subspace (typically k << d), the first principal component
    will have the largest possible variance. All consequent principal
    components will have the largest variance given the constraint that these
    components are uncorrelated (orthogonal) to the other principal
    components—even if the input features are correlated, the resulting
    principal components will be mutually orthogonal (uncorrelated). Note that
    the PCA directions are highly sensitive to data scaling, and we need to
    standardize the features prior to PCA if the features were measured on
    different scales and we want to assign equal importance to all features.

    Before looking at the PCA algorithm for dimensionality reduction in more
    detail, let's summarize the approach in a few simple steps:

        1. Standardize the d-dimensional dataset.
        2. Construct the covariance matrix.
        3. Decompose the covariance matrix into its eigenvectors and
           eigenvalues.
        4. Sort the eigenvalues by decreasing order to rank the corresponding
           eigenvectors.
        5. Select k eigenvectors, which correspond to the k largest
           eigenvalues, where k is the dimensionality of the new feature
           subspace ( k ≤ d ).
        6. Construct a projection matrix, W, from the "top" k eigenvectors.
        7. Transform the d-dimensional input dataset, X, using the projection
        matrix, W, to obtain the new k-dimensional feature subspace.

"""
