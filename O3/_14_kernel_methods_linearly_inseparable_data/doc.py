"""
    Obviously, we would not be able to separate the examples from the positive
    and negative class very well using a linear hyperplane as a decision
    boundary via the linear logistic regression or linear SVM model.

    The basic idea behind kernel methods to deal with such linearly inseparable
    data is to create nonlinear combinations of the original features to
    project them onto a higher-dimensional space via a mapping function, φ,
    where the data becomes linearly separable.

    To solve a nonlinear problem using an SVM, we would transform the training
    data into a higher-dimensional feature space via a mapping function, φ ,
    and train a linear SVM model to classify the data in this new feature
    space. Then, we could use the same mapping function φ, to transform
    new, unseen data to classify it, using the linear SVM model.

"""
