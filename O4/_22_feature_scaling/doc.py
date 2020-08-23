"""

    Feature scaling is a crucial step in our preprocessing pipeline that can
    easily be forgotten. Decision trees and random forests are two of the very
    few machine learning algorithms where we don't need to worry about feature
    scaling. Those algorithms are scale invariant. However, the majority of
    machine learning and optimization algorithms behave much better if
    features are on the same scale.

    The importance of feature scaling can be illustrated by a simple example.
    Let's assume that we have two features where one feature is measured on a
    scale from 1 to 10 and the second feature is measured on a scale from 1 to
    100,000, respectively. When we think of the squared error function in
    Adaline, it makes sense to say that the algorithm will mostly be busy
    optimizing the weights according to the larger errors in the second
    feature. Another example is the k-nearest neighbors (KNN) algorithm with a
    Euclidean distance measure: the computed distances between examples will
    be dominated by the second feature axis.

    Now, there are two common approaches to bringing different features onto
    the same scale: normalization and standardization.
    Those terms are often used quite loosely in different fields, and the
    meaning has to be derived from the context.

    Most often, normalization refers to the rescaling of the features to a
    range of [0, 1], which is a special case of min-max scaling.

        Normalization: x(i norm) = (x(i) - x(min)) / (x(max) - x(min))


    Although normalization via min-max scaling is a commonly used technique
    that is useful when we need values in a bounded interval, standardization
    can be more practical for many machine learning algorithms, especially for
    optimization algorithms such as gradient descent. The reason is that many
    linear models, such as the logistic regression and SVM, initialize the
    weights to 0 or small random values close to 0. Using standardization, we
    center the feature columns at mean 0 with standard deviation 1 so that the
    feature columns have the same parameters as a standard normal distribution
    (zero mean and unit variance), which makes it easier to learn the weights.
    Furthermore, standardization maintains useful information about outliers
    and makes the algorithm less sensitive to them in contrast to min-max
    scaling, which scales the data to a limited range of values.

        Standardization: x(i std) = (x(i) - μ(x)) / σ(x)

    Here, μ(x) is the sample mean of a particular feature column, and σ(x) is
    the corresponding standard deviation.

"""
