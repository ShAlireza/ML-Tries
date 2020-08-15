from typing import Tuple

import numpy as np


class AdalineMBGD:
    """ADAptive LInear NEuron classifier (Mini Batch).

    Parameters
    ----------
    eta : float
        Learning rate (between 0.0 and 1.0)
    epochs : int
        Passes over the training dataset
    random_state : int
        Random number generator seed for random weight
        initialization.
    activation : function
        Optional activation function provided by user
    batch_size : int
        Count of training data used in every epoch

    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting.
    cost_ : list
        Sum of squares (SSE) cost (loss) function value in each epoch.

    """

    def __init__(self, eta=0.01, epochs=50, random_state=1, activation=None,
                 batch_size=50):
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
        self.batch_size = batch_size
        self._activation = activation
        self.w_ = None
        self.random_generator = np.random.RandomState(seed=random_state)
        self.cost_ = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> 'AdalineMBGD':
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_example is the number of
            examples and n_features is the number of features.

        Y : array-like, shape = [n_examples]
            True labels.

        Returns
        -------
        self : AdalineGD

        """
        random_generator = np.random.RandomState(seed=self.random_state)
        self.w_ = random_generator.normal(loc=0.0, scale=0.01,
                                          size=1 + X.shape[1])

        self.cost_ = []

        for _ in range(self.epochs):
            X_m, Y_m = self._mini_batch(X, Y)
            net_input = self.net_input(X_m)
            output = self.activation(net_input)
            errors = Y_m - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2
            self.cost_.append(cost)
        return self

    def _mini_batch(self, X: np.ndarray,
                    Y: np.ndarray) -> 'Tuple[np.ndarray, np.ndarray]':
        """Get a random mini batch from complete data"""
        random_indexes = self.random_generator.choice(np.arange(0, Y.size),
                                                      size=self.batch_size)
        return X[random_indexes], Y[random_indexes]

    def _initialize_weights(self, size: int) -> 'None':
        """Initialize weights to small random numbers"""
        self.w_ = self.random_generator.normal(loc=0.0, scale=0.01,
                                               size=1 + size)

    def activation(self, X: np.ndarray) -> 'np.ndarray':
        """Apply linear activation function"""
        return self._activation(X) if self._activation else X

    def net_input(self, X: np.ndarray) -> 'np.ndarray':
        """Calculate net input"""
        return X.dot(self.w_[1:]) + self.w_[0]

    def predict(self, X: np.ndarray) -> 'np.ndarray':
        """Return class label prediction"""
        return np.where(self.net_input(X) > 0, 1, -1)
