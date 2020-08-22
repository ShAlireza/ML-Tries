"""

    When we are talking about categorical data, we have to further distinguish
    between ordinal and nominal features. Ordinal features can be understood as
    categorical values that can be sorted or ordered. For example, t-shirt size
    would be an ordinal feature, because we can define an order: XL > L > M. In
    contrast, nominal features don't imply any order and, to continue with the
    previous example, we could think of t-shirt color as a nominal feature
    since it typically doesn't make sense to say that, for example, red is
    larger than blue.

    Mapping Ordinal features:

    To make sure that the learning algorithm interprets the ordinal features
    correctly, we need to convert the categorical string values into integers.
    Unfortunately, there is no convenient function that can automatically
    derive the correct order of the labels of our size feature, so we have to
    define the mapping manually.



"""
