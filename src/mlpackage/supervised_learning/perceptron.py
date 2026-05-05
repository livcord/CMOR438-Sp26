import numpy as np


class Perceptron:
    """
    Binary classifier using the Perceptron learning rule.

    Improvements:
    - Stable convergence tracking (post-epoch evaluation)
    - Random weight initialization (avoids symmetry issues)
    - Deterministic behavior via random_state
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000, random_state=42):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if n_iterations <= 0:
            raise ValueError("n_iterations must be a positive integer.")

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state

        self.weights = None
        self.bias = None
        self.errors_per_epoch = []

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)

        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features).")
        if X.shape[0] != len(y):
            raise ValueError("X and y must have the same number of samples.")
        if len(np.unique(y)) > 2:
            raise ValueError("Perceptron only supports binary classification.")

        # Convert labels to {-1, +1}
        y_internal = np.where(y == 0, -1, 1)

        n_samples, n_features = X.shape

        rng = np.random.default_rng(self.random_state)

        # Random initialization (break symmetry)
        self.weights = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias = 0.0

        self.errors_per_epoch = []

        for _ in range(self.n_iterations):
            # Shuffle each epoch
            indices = rng.permutation(n_samples)

            for i in indices:
                x_i = X[i]
                y_i = y_internal[i]

                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = 1 if linear_output >= 0 else -1

                if y_i != y_pred:
                    self.weights += self.learning_rate * y_i * x_i
                    self.bias += self.learning_rate * y_i

            # Compute errors after epoch (true convergence signal)
            linear_output_all = np.dot(X, self.weights) + self.bias
            y_pred_all = np.where(linear_output_all >= 0, 1, -1)

            errors = np.sum(y_pred_all != y_internal)
            self.errors_per_epoch.append(errors)

            # Early stopping
            if errors == 0:
                break

        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)

    def score(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)
        return np.mean(self.predict(X) == y)