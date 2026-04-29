import numpy as np


class Perceptron:
    """
    Binary classifier using the Perceptron learning rule.

    The Perceptron is a single-layer linear classifier that updates weights
    whenever a sample is misclassified. It converges only if the data is
    linearly separable.

    Parameters
    ----------
    learning_rate : float
        Step size for weight updates. Default is 0.01.
    n_iterations : int
        Maximum number of passes over the training data. Default is 1000.

    Attributes
    ----------
    weights : np.ndarray of shape (n_features,)
        Learned weight vector after fitting.
    bias : float
        Learned bias term after fitting.
    errors_per_epoch : list of int
        Number of misclassifications per training epoch (useful for plotting).
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if n_iterations <= 0:
            raise ValueError("n_iterations must be a positive integer.")
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.errors_per_epoch = []

    def fit(self, X, y):
        """
        Train the Perceptron on labeled data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Binary labels. Must contain only 0 and 1.

        Returns
        -------
        self
        """
        X = np.array(X, dtype=float)
        y = np.array(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if X.shape[0] != len(y):
            raise ValueError(
            "X and y must have the same number of samples,"
            f"got X: {X.shape[0]}, y: {len(y)}."
            )
        if len(np.unique(y)) > 2:
            raise ValueError("Perceptron only supports binary classification.")

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.errors_per_epoch = []

        for _ in range(self.n_iterations):
            errors = 0
            for x_i, y_i in zip(X, y):
                y_pred = self._predict_single(x_i)
                update = self.learning_rate * (y_i - y_pred)
                self.weights += update * x_i
                self.bias += update
                errors += int(update != 0)
            self.errors_per_epoch.append(errors)
            # Early stopping: if no errors, the data is already separated
            if errors == 0:
                break

        return self

    def predict(self, X):
        """
        Predict binary class labels for samples in X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,) with values 0 or 1.
        """
        X = np.array(X, dtype=float)
        return np.array([self._predict_single(x) for x in X])

    def score(self, X, y):
        """
        Return classification accuracy on (X, y).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)

        Returns
        -------
        float : fraction of correctly classified samples.
        """
        return np.mean(self.predict(X) == np.array(y))

    def _predict_single(self, x):
        """Apply step activation to a single sample."""
        return 1 if np.dot(x, self.weights) + self.bias >= 0 else 0