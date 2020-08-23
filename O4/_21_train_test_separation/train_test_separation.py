import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from O4 import prepare_data

X, y = prepare_data(head=5, print_classes=True)

# Providing the class label array y as an argument to stratify ensures that
# both training and test datasets have the same class proportions as the
# original dataset.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0,
                                                    stratify=y)

# If we are dividing a dataset into training and test datasets,
# we have to keep in mind that we are withholding valuable
# information that the learning algorithm could benefit from.
# Thus, we don't want to allocate too much information to the test
# set. However, the smaller the test set, the more inaccurate the
# estimation of the generalization error. Dividing a dataset into
# training and test datasets is all about balancing this tradeoff. In
# practice, the most commonly used splits are 60:40, 70:30, or 80:20,
# depending on the size of the initial dataset. However, for large
# datasets, 90:10 or 99:1 splits are also common and appropriate.
# For example, if the dataset contains more than 100,000 training
# examples, it might be fine to withhold only 10,000 examples
# for testing in order to get a good estimate of the generalization
# performance. More information and illustrations can be found
# in section one of this  article Model evaluation, model selection, and
# algorithm selection in machine learning, which is freely available
# at https://arxiv.org/pdf/1811.12808.pdf .
