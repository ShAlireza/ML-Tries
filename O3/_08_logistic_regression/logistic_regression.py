import numpy as np


class LogisticRegressionGD:
    """Logistic Regression classifier.

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
    cost_ : list
        Logistic cost function value in each epoch.

    """

    def __init__(self, eta=0.01, epochs=50, random_state=1):
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
        self.w_ = None
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
        self : LogisticRegressionGD

        """
        random_generator = np.random.RandomState(seed=self.random_state)
        self.w_ = random_generator.normal(loc=0.0, scale=0.01,
                                          size=1 + X.shape[1])

        self.cost_ = []

        for _ in range(self.epochs):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (Y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (-Y.dot(np.log(output)) -
                    (1 - Y).dot(np.log(1 - output)))
            self.cost_.append(cost)

        return self

    def activation(self, X: np.ndarray):
        """Apply logistic sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(X, a_min=-250, a_max=250)))

    def net_input(self, X: np.ndarray):
        """Calculate net input"""
        return X.dot(self.w_[1:]) + self.w_[0]

    def predict(self, X: np.ndarray):
        """Return class label prediction"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

        # equivalent to :
        #   return np.where(self.net_input(X) >= 0, 1, 0)
