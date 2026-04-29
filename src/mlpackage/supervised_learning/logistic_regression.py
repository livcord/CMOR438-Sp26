import numpy as np

class LogisticRegression:
    """
    Binary Logistic Regression via batch gradient descent.

    Maps a linear combination of features through the sigmoid function to
    produce class probabilities, then minimizes binary cross-entropy loss.
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000, threshold=0.5):
        if not (0 < threshold < 1):
            raise ValueError("threshold must be strictly between 0 and 1.")

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.threshold = threshold

        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        """
        Fit logistic regression to training data.
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        # Validation
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if y.ndim != 1:
            raise ValueError("y must be 1D.")
        if X.shape[0] == 0:
            raise ValueError("Cannot fit on empty dataset.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        unique = np.unique(y)
        if not np.all(np.isin(unique, [0, 1])):
            raise ValueError("y must contain only binary labels {0, 1}.")

        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for i in range(self.n_iterations):
            linear_output = X @ self.weights + self.bias
            y_prob = self._sigmoid(linear_output)

            # Gradients
            error = y_prob - y
            dw = (X.T @ error) / n_samples
            db = np.sum(error) / n_samples

            # Update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Track loss
            loss = self._binary_cross_entropy(y, y_prob)
            self.loss_history.append(loss)

            # Optional early stopping (small improvement over original)
            if i > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < 1e-7:
                break

        return self

    def predict_proba(self, X):
        """
        Return P(y=1 | X) for each sample.
        """
        if self.weights is None:
            raise TypeError("Model must be fitted before calling predict_proba.")

        X = np.array(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if X.shape[1] != self.weights.shape[0]:
            raise ValueError("Feature dimension mismatch.")

        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X):
        """
        Predict binary class labels.
        """
        probs = self.predict_proba(X)
        return (probs >= self.threshold).astype(int)

    def score(self, X, y):
        """
        Return classification accuracy.
        """
        y = np.array(y)
        preds = self.predict(X)
        return np.mean(preds == y)

    # Internal helpers
    @staticmethod
    def _sigmoid(z):
        """Numerically stable sigmoid."""
        return np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),
            np.exp(z) / (1 + np.exp(z))
        )

    @staticmethod
    def _binary_cross_entropy(y_true, y_prob):
        """Binary cross-entropy loss."""
        eps = 1e-15
        y_prob = np.clip(y_prob, eps, 1 - eps)
        return -np.mean(
            y_true * np.log(y_prob) +
            (1 - y_true) * np.log(1 - y_prob)
        )