"""

    If we notice that a model performs much better on a training dataset than
    on the test dataset, this observation is a strong indicator of overfitting.
    the model fits the parameters too closely with regard to the particular
    observations in the training dataset, but does not generalize well to new
    data; we say that the model has a high variance.
    The reason for the overfitting is that our model is too complex for the
    given training data. Common solutions to reduce the generalization
    error are as follows:
        • Collect more training data
        • Introduce a penalty for complexity via regularization
        • Choose a simpler model with fewer parameters
        • Reduce the dimensionality of the data

    Collecting more training data is often not applicable. In the following
    sections, we will look at common ways to reduce overfitting by
    regularization and dimensionality reduction via feature selection, which
    leads to simpler models by requiring fewer parameters to be fitted to the
    data.

"""
