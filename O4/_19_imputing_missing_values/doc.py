"""

    Often, the removal of training examples or dropping of entire feature
    columns is simply not feasible, because we might lose too much valuable
    data. In this case, we can use different interpolation techniques to
    estimate the missing values from the other training examples in our
    dataset. One of the most common interpolation techniques is mean
    imputation, where we simply replace the missing value with the mean value
    of the entire feature column. A convenient way to achieve this is by using
    the SimpleImputer class from scikit-learn.

"""
