"""Dealing with missing data

    It is not uncommon in real-world applications for our training examples to
    be missing one or more values for various reasons. There could have been
    an error in the data collection process, certain measurements may not be
    applicable, or particular fields could have been simply left blank in a
    survey, for example. We typically see missing values as blank spaces in our
    data table or as placeholder strings such as NaN , which stands for "not a
    number," or NULL (a commonly used indicator of unknown values in relational
    databases). Unfortunately, most computational tools are unable to handle
    such missing values or will produce unpredictable results if we simply
    ignore them. Therefore, it is crucial that we take care of those missing
    values before we proceed with further analyses.

"""
