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

    The main advantage of such a memory-based approach is that the classifier
    immediately adapts as we collect new training data. However, the downside
    is that the computational complexity for classifying new examples grows
    linearly with the number of examples in the training dataset in the
    worst-case scenario—unless the dataset has very few dimensions (features)
    and the algorithm has been implemented using efficient data structures such
    as k-d trees (An Algorithm for Finding Best Matches in Logarithmic Expected
    Time, J. H. Friedman, J. L. Bentley, and R.A. Finkel, ACM transactions on
    mathematical software (TOMS), 3(3): 209–226, 1977). Furthermore, we can't
    discard training examples since no training step is involved. Thus, storage
    space can become a challenge if we are working with large datasets.

"""
