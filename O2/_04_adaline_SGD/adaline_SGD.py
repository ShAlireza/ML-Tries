from typing import Tuple, Union

import numpy as np


class AdalineSGD:
    """ADAptive LInear NEuron classifier.

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
    shuffle : bool
        Shuffles training data every epoch if True to prevent
        cycles.

    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting.
    cost_ : list
        Sum of squares (SSE) cost (loss) function value in each epoch.

    """

    def __init__(self, eta=0.01, epochs=50, random_state=1, activation=None,
                 shuffle=True):
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
        self.shuffle = shuffle
        self._activation = activation
        self.w_ = None
        self.w_initialized = False
        self.random_generator = None
        self.cost_ = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
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
        self : AdalineSGD

        """
        self._initialize_weights(X.shape[0])

        self.cost_ = []

        for _ in range(self.epochs):
            if self.shuffle:
                X, y = self._shuffle(X, Y)
            cost = 0
            for x, y in zip(X, Y):
                cost += self._update_weights(x, y)
            avg_cost = cost / Y.size
            self.cost_.append(avg_cost)

        return self

    def partial_fit(self, X: np.ndarray, Y: np.ndarray) -> 'AdalineSGD':
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if Y.ravel().shape[0] > 1:
            for x, y in zip(X, Y):
                self._update_weights(x, y)
        else:
            self._update_weights(X, Y)
        return self

    def _shuffle(self, X: np.ndarray,
                 Y: np.ndarray) -> 'Tuple[np.ndarray, np.ndarray]':
        """Shuffle training data"""
        indexes = self.random_generator.permutation(Y.size)
        return X[indexes], Y[indexes]

    def _initialize_weights(self, size: int) -> 'None':
        """Initialize weights to small random numbers"""
        self.random_generator = np.random.RandomState(seed=self.random_state)
        self.w_ = self.random_generator.normal(loc=0.0, scale=0.01,
                                               size=1 + size)
        self.w_initialized = True

    def _update_weights(self, x: np.ndarray,
                        target: Union[float, np.ndarray]) -> 'float':
        """Apply Adaline learning rule to update weights"""
        output = self.activation(self.net_input(x))
        error = target - output
        self.w_[1:] = self.eta * x.dot(error)
        self.w_[0] = self.eta * error
        cost = error ** 2 / 2.0
        return cost

    def activation(self, X: np.ndarray) -> 'np.ndarray':
        """Apply linear activation function"""
        return self._activation(X) if self._activation else X

    def net_input(self, X: np.ndarray) -> 'np.ndarray':
        """Calculate net input"""
        return X.dot(self.w_[1:]) + self.w_[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class label prediction"""
        return np.where(self.net_input(X) > 0, 1, -1)
