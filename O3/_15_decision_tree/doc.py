"""Decision Tree And common Impurity Functions

    Based on the features in our training dataset, the decision tree model
    learns a series of questions to infer the class labels of the examples.

    Using the decision algorithm, we start at the tree root and split the data
    on the feature that results in the largest information gain (IG).

    In order to split the nodes at the most informative features, we need to
    define an objective function that we want to optimize via the tree learning
    algorithm.

    The three impurity measures or splitting criteria that are commonly used in
    binary decision trees are Gini impurity ( I-G ), entropy ( I-H ), and
    the classification error ( I-E ).

    However, in practice, both Gini impurity and entropy typically yield very
    similar results, and it is often not worth spending much time on evaluating
    trees using different impurity criteria rather than experimenting with
    different pruning cut-offs.

    Classification Error is a useful criterion for pruning but not recommended
    for growing a decision tree, since it is less sensitive to changes in the
    class probabilities of the nodes.

"""
