"""
    A random forest can be
    considered as an ensemble of decision trees. The idea behind a random
    forest is to average multiple (deep) decision trees that individually
    suffer from high variance to build a more robust model that has a better
    generalization performance and is less susceptible to overfitting. The
    random forest algorithm can be summarized in four simple steps:

        1. Draw a random bootstrap sample of size n (randomly choose n examples
        from the training dataset with replacement).
        2. Grow a decision tree from the bootstrap sample. At each node:
            a. Randomly select d features without replacement.
            b. Split the node using the feature that provides the best split
            according to the objective function, for instance, maximizing the
            information gain.
        3. Repeat the steps 1-2 k times.
        4. Aggregate the prediction by each tree to assign the class label by
        majority vote.

    Although random forests don't offer the same level of interpretability as
    decision trees, a big advantage of random forests is that we don't have to
    worry so much about choosing good hyperparameter values. We typically don't
    need to prune the random forest since the ensemble model is quite robust to
    noise from the individual decision trees. The only parameter that we really
    need to care about in practice is the number of trees, k, (step 3) that we
    choose for the random forest. Typically, the larger the number of trees,
    the better the performance of the random forest classifier at the
    expense of an increased computational cost.

    Although it is less common in practice, other hyperparameters of the random
    forest classifier that can be optimized are the size, n, of the bootstrap
    sample (step 1), and the number of features, d, that are randomly chosen
    for each split (step 2.a), respectively. Via the sample size, n, of the
    bootstrap sample, we control the bias-variance tradeoff of the random
    forest.

    Decreasing the size of the bootstrap sample increases the diversity among
    the individual trees, since the probability that a particular training
    example is included in the bootstrap sample is lower. Thus, shrinking the
    size of the bootstrap samples may increase the randomness of the random
    forest, and it can help to reduce the effect of overfitting. However,
    smaller bootstrap samples typically result in a lower overall performance
    of the random forest, and a small gap between training and test
    performance, but a low test performance overall. Conversely, increasing the
    size of the bootstrap sample may increase the degree of overfitting.
    Because the bootstrap samples, and consequently the individual decision
    trees, become more similar to each other, they learn to fit the original
    training dataset more closely.

    In most implementations, including the RandomForestClassifier
    implementation in scikit-learn, the size of the bootstrap sample is chosen
    to be equal to the number of training examples in the original training
    dataset, which usually provides a good bias-variance tradeoff. For the
    number of features, d, at each split, we want to choose a value that is
    smaller than the total number of features in the training dataset. A
    reasonable default that is used in scikit-learn and other implementations
    is d = √m , where m is the number of features in the training dataset.

"""
