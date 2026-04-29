import numpy as np

class LinearRegression:
    """
    Ordinary Least Squares Linear Regression.

    Supports two fitting methods:
      - 'ols'  : closed-form Normal Equation (fast, exact, but O(n³) in features)
      - 'gd'   : batch gradient descent (scalable, iterative)

    Parameters
    method : str
        'ols' or 'gd'. Default is 'ols'.
    learning_rate : float
        Step size for gradient descent. Ignored when method='ols'. Default 0.01.
    n_iterations : int
        Number of gradient descent steps. Ignored when method='ols'. Default 1000.

    Attributes
    weights : np.ndarray of shape (n_features,)
    bias : float
    loss_history : list of float
        MSE at each GD iteration. Empty when method='ols'.
    """

    def __init__(self, method="ols", learning_rate=0.01, n_iterations=1000):
        if method not in ("ols", "gd"):
            raise ValueError("method must be 'ols' or 'gd'.")
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        """
        Fit the model to training data.

        Parameters
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)

        Returns
        self
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        if self.method == "ols":
            self._fit_ols(X, y)
        else:
            self._fit_gd(X, y)
        return self

    def _fit_ols(self, X, y):
        """Normal equation: w = (XᵀX)⁻¹ Xᵀy (with bias column prepended)."""
        X_b = np.column_stack([np.ones(X.shape[0]), X])   # add bias column
        # Use lstsq for numerical stability instead of explicit inverse
        theta, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
        self.bias = theta[0]
        self.weights = theta[1:]

    def _fit_gd(self, X, y):
        """Batch gradient descent."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iterations):
            y_pred = self._linear(X)
            error = y_pred - y

            dw = (1 / n_samples) * X.T @ error
            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            mse = np.mean(error ** 2)
            self.loss_history.append(mse)

    def predict(self, X):
        """
        Predict continuous target values.

        Parameters
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        np.ndarray of shape (n_samples,)
        """
        return self._linear(np.array(X, dtype=float))

    def _linear(self, X):
        return X @ self.weights + self.bias

    # Evaluation metrics 

    def mse(self, X, y):
        """Mean Squared Error."""
        return np.mean((self.predict(X) - np.array(y)) ** 2)

    def rmse(self, X, y):
        """Root Mean Squared Error."""
        return np.sqrt(self.mse(X, y))

    def r_squared(self, X, y):
        """
        Coefficient of determination R².
        R² = 1 means perfect fit; R² = 0 means the model is no better than
        predicting the mean.
        """
        y = np.array(y, dtype=float)
        ss_res = np.sum((y - self.predict(X)) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0