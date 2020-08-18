"""Regularization and feature normalization

    Regularization is another reason why feature scaling such as
    standardization is important. For regularization to work properly,
    we need to ensure that all our features are on comparable scales.

    Via the regularization parameter, λ , we can then control how well
    we fit the training
    data, while keeping the weights small. By increasing the value of λλ ,
    we increase the regularization strength.
    The parameter, C , that is implemented for the LogisticRegression class
    in scikit-learn comes from a convention in support vector machines,
    which will be the topic of the next section. The term C is directly
    related to the regularization parameter, λ , which is its inverse.
    Consequently, decreasing the value of the inverse regularization parameter,
    C , means that we are increasing the regularization strength.

"""
