import numpy as np


class Perceptron:
    """Perceptron classifier.

    Parameters
    ----------
    eta : float
        Learning rate (between 0.0 and 1.0)
    epochs : int
        Passes over the training dataset
    random_state : int
        Random number generator seed for random weight
        initialization.

    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of miss_classifications (updates) in each epoch.

    """

    def __init__(self, eta=0.01, epochs=50, random_state=1):
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
        self.w_ = None
        self.errors_ = None

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
        self : Perceptron

        """
        random_generator = np.random.RandomState(seed=self.random_state)
        self.w_ = random_generator.normal(loc=0.0, scale=0.01,
                                          size=1 + X.shape[1])

        self.errors_ = []

        for _ in range(self.epochs):
            error = 0
            for x, y in zip(X, Y):
                predicted_y = self.predict(x)
                update = (y - predicted_y) * self.eta
                self.w_[1:] += update * x
                self.w_[0] = self.w_[0] + update

                error += int(update != 0.0)
            self.errors_.append(error)
        return self

    def net_input(self, x: np.ndarray):
        """Calculate net input"""
        return x.dot(self.w_[1:]) + self.w_[0]

    def predict(self, x: np.ndarray):
        """Return class label prediction"""
        return np.where(self.net_input(x) >= 0, 1, -1)
