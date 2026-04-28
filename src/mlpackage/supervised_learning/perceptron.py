import numpy as np

class Perceptron:
    """
    Binary classifier using the Perceptron learning rule.

    Parameters
    learning_rate : float
        Step size for weight updates (default 0.01).
    n_iterations : int
        Number of passes over the training data (default 1000).
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """Train on feature matrix X and binary labels y."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                prediction = self.predict_single(x_i)
                update = self.learning_rate * (y[idx] - prediction)
                self.weights += update * x_i
                self.bias += update

    def predict_single(self, x):
        return 1 if np.dot(x, self.weights) + self.bias >= 0 else 0

    def predict(self, X):
        """Return predicted labels for feature matrix X."""
        return np.array([self.predict_single(x) for x in X])