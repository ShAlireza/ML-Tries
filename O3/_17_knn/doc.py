"""

    KNN is a typical example of a lazy learner. It is called "lazy" not because
    of its apparent simplicity, but because it doesn't learn a discriminative
    function from the training data but memorizes the training dataset instead.

    The KNN algorithm itself is fairly straightforward and can be summarized by
    the following steps:
    1. Choose the number of k and a distance metric.
    2. Find the k-nearest neighbors of the data record that we want to classify.
    3. Assign the class label by majority vote.

    Based on the chosen distance metric, the KNN algorithm finds the k examples
    in the training dataset that are closest (most similar) to the point that
    we want to classify. The class label of the data point is then determined
    by a majority vote among its k nearest neighbors.

"""
