"""

    An alternative way to reduce the complexity of the model and avoid
    overfitting is dimensionality reduction via feature selection, which is
    especially useful for unregularized models. There are two main categories
    of dimensionality reduction techniques: feature selection and feature
    extraction. Via feature selection, we select a subset of the original
    features, whereas in feature extraction, we derive information
    from the feature set to construct a new feature subspace.

    In this section, we will take a look at a classic family of
    feature selection algorithms

    Sequential feature selection algorithms are a family of greedy search
    algorithms that are used to reduce an initial d-dimensional feature space
    to a k-dimensional feature subspace where k<d. The motivation behind
    feature selection algorithms is to automatically select a subset of
    features that are most relevant to the problem, to improve computational
    efficiency, or to reduce the generalization error of the model by removing
    irrelevant features or noise, which can be useful for algorithms that
    don't support regularization.



    A classic sequential feature selection algorithm is sequential backward 
    selection (SBS), which aims to reduce the dimensionality of the initial 
    feature subspace with a minimum decay in the performance of the classifier 
    to improve upon computational efficiency. In certain cases, SBS can even 
    improve the predictive power of the model if a model suffers from 
    overfitting.
    
    The idea behind the SBS algorithm is quite simple: SBS sequentially removes 
    features from the full feature subset until the new feature subspace 
    contains the desired number of features. In order to determine which 
    feature is to be removed at each stage, we need to define the criterion 
    function, J, that we want to minimize.
    
    The criterion calculated by the criterion function can simply be the 
    difference in performance of the classifier before and after the removal 
    of a particular feature. Then, the feature to be removed at each stage can 
    simply be defined as the feature that maximizes this criterion; or in more 
    simple terms, at each stage we eliminate the feature that causes the least 
    performance loss after removal. Based on the preceding definition of SBS, 
    we can outline the algorithm in four simple steps:
    
        1. Initialize the algorithm with k = d, where d is the dimensionality 
        of the full feature space, X(d).
        2. Determine the feature, x(,-) , that maximizes the criterion: x(-) = 
        argmax J(X(k) − x) , where x ∈ X(k).
        3. Remove the feature, x(,-), from the feature set: 
        X(k−1) = X(k) − x(,-); k = k − 1.
        4. Terminate if k equals the number of desired features; otherwise, go 
        to step 2.

"""

from itertools import combinations

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class SBS(object):
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = estimator
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, X_test, y_test,
                                 self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train, X_test,
                                         y_test, p)
                scores.append(score)
                subsets.append(p)

            best = int(np.argmax(scores))
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])

        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_true=y_test, y_pred=y_pred)
        return score
