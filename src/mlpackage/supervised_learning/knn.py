import numpy as np
from collections import Counter

class KNN:
    """
    K-Nearest Neighbors classifier and regressor.

    Predicts by finding the K closest training samples to a query point
    (using a chosen distance metric) and aggregating their labels.

    Parameters
    k : int
        Number of neighbors to consider. Default is 5.
    task : str
        'classification' returns the majority vote label.
        'regression' returns the mean of neighbor values.
        Default is 'classification'.
    metric : str
        Distance metric: 'euclidean', 'manhattan', or 'minkowski'.
        Default is 'euclidean'.
    p : int
        Power parameter for Minkowski distance. Ignored unless
        metric='minkowski'. Default is 3.

    Attributes
    X_train : np.ndarray of shape (n_samples, n_features)
    y_train : np.ndarray of shape (n_samples,)
    """

    VALID_METRICS = ("euclidean", "manhattan", "minkowski")
    VALID_TASKS = ("classification", "regression")

    def __init__(self, k=5, task="classification", metric="euclidean", p=3):
        if k < 1:
            raise ValueError("k must be >= 1.")
        if task not in self.VALID_TASKS:
            raise ValueError(f"task must be one of {self.VALID_TASKS}.")
        if metric not in self.VALID_METRICS:
            raise ValueError(f"metric must be one of {self.VALID_METRICS}.")
        self.k = k
        self.task = task
        self.metric = metric
        self.p = p
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Store training data. KNN is a lazy learner — no computation happens here.

        Parameters
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)

        Returns
        self
        """
        X = np.array(X, dtype=float)
        y = np.array(y)

        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if X.shape[0] != len(y):
            raise ValueError("X and y must have the same number of samples.")
        if self.k > X.shape[0]:
            raise ValueError(f"k ({self.k}) cannot exceed number of training samples ({X.shape[0]}).")

        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        """
        Predict labels or values for each sample in X.

        Parameters
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        np.ndarray of shape (n_samples,)
        """
        if self.X_train is None:
            raise AttributeError("Model must be fitted before calling predict.")

        X = np.array(X, dtype=float)
        return np.array([self._predict_single(x) for x in X])

    def score(self, X, y):
        """
        Accuracy (classification) or R² (regression) on (X, y).
        """
        y = np.array(y)
        preds = self.predict(X)
        if self.task == "classification":
            return np.mean(preds == y)
        else:
            ss_res = np.sum((y - preds) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    # Internal helpers 

    def _predict_single(self, x):
        """Find k nearest neighbors and aggregate their labels."""
        distances = self._compute_distances(x)
        k_indices = np.argsort(distances)[: self.k]
        k_labels = self.y_train[k_indices]

        if self.task == "classification":
            return Counter(k_labels).most_common(1)[0][0]
        else:
            return np.mean(k_labels)

    def _compute_distances(self, x):
        """Compute distance from x to every training sample."""
        if self.metric == "euclidean":
            return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        elif self.metric == "manhattan":
            return np.sum(np.abs(self.X_train - x), axis=1)
        else:  # minkowski
            return np.sum(np.abs(self.X_train - x) ** self.p, axis=1) ** (1 / self.p)