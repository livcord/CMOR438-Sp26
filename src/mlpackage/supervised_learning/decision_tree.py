import numpy as np
from collections import Counter


class _Node:
    """Internal node of the decision tree."""

    def __init__(self, feature=None, threshold=None, left=None,
                 right=None, *, value=None):
        self.feature = feature      # feature index to split on
        self.threshold = threshold  # split threshold value
        self.left = left            # left subtree  (feature <= threshold)
        self.right = right          # right subtree (feature >  threshold)
        self.value = value          # leaf prediction (set only at leaves)

    @property
    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    """
    Decision Tree for classification and regression, built with recursive
    binary splitting.

    Splitting criterion:
      - Classification : Gini impurity  (task='classification')
      - Regression     : Variance reduction (task='regression')

    Parameters
    task : str
        'classification' or 'regression'. Default is 'classification'.
    max_depth : int or None
        Maximum depth of the tree. None means nodes expand until leaves
        are pure or contain fewer than min_samples_split samples.
        Default is None.
    min_samples_split : int
        Minimum samples required to split a node. Default is 2.
    min_samples_leaf : int
        Minimum samples required to be at a leaf node. Default is 1.
    n_features : int or None
        Number of features to consider at each split (for use inside
        Random Forests). None means all features. Default is None.

    Attributes
    root : _Node
        Root node of the fitted tree.
    n_features_in_ : int
        Number of features seen during fit.
    """

    VALID_TASKS = ("classification", "regression")

    def __init__(self, task="classification", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, n_features=None):
        if task not in self.VALID_TASKS:
            raise ValueError(f"task must be one of {self.VALID_TASKS}.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2.")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1.")

        self.task = task
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_features = n_features
        self.root = None
        self.n_features_in_ = None


    def fit(self, X, y):
        """
        Build the decision tree from training data.

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
        if X.shape[0] == 0:
            raise ValueError("Cannot fit on empty dataset.")

        self.n_features_in_ = X.shape[1]
        # Determine how many features to sample at each split
        self._n_features_split = (
            self.n_features if self.n_features is not None
            else X.shape[1]
        )
        self.root = self._grow(X, y, depth=0)
        return self

    def predict(self, X):
        """
        Predict labels or values for samples in X.

        Parameters
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        np.ndarray of shape (n_samples,)
        """
        if self.root is None:
            raise AttributeError("Model must be fitted before calling predict.")
        X = np.array(X, dtype=float)
        return np.array([self._traverse(x, self.root) for x in X])

    def score(self, X, y):
        """Accuracy (classification) or R² (regression)."""
        y = np.array(y)
        preds = self.predict(X)
        if self.task == "classification":
            return np.mean(preds == y)
        else:
            ss_res = np.sum((y - preds) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    # Tree building

    def _grow(self, X, y, depth):
        """Recursively grow the tree by finding the best split."""
        n_samples, n_features = X.shape

        # Stopping conditions → create a leaf
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or self._is_pure(y)
        ):
            return _Node(value=self._leaf_value(y))

        # Find best split 
        feature_indices = np.random.choice(
            n_features, self._n_features_split, replace=False
        )
        best_feat, best_thresh = self._best_split(X, y, feature_indices)

        if best_feat is None:           # no valid split found
            return _Node(value=self._leaf_value(y))

        # Partition and recurse
        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        # Enforce min_samples_leaf
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            return _Node(value=self._leaf_value(y))

        left = self._grow(X[left_mask], y[left_mask], depth + 1)
        right = self._grow(X[right_mask], y[right_mask], depth + 1)
        return _Node(feature=best_feat, threshold=best_thresh,
                     left=left, right=right)

    def _best_split(self, X, y, feature_indices):
        """Search over candidate features and thresholds for the best split."""
        best_gain = -np.inf
        best_feat, best_thresh = None, None

        for feat in feature_indices:
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                gain = self._information_gain(y, X[:, feat], thresh)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        return best_feat, best_thresh

    def _information_gain(self, y, col, threshold):
        """
        Compute the reduction in impurity from splitting col on threshold.
        Uses Gini for classification, variance for regression.
        """
        left_mask = col <= threshold
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return -np.inf

        n = len(y)
        n_l, n_r = left_mask.sum(), right_mask.sum()

        parent_impurity = self._impurity(y)
        child_impurity = (n_l / n) * self._impurity(y[left_mask]) \
                       + (n_r / n) * self._impurity(y[right_mask])

        return parent_impurity - child_impurity

    def _impurity(self, y):
        """Gini impurity (classification) or variance (regression)."""
        if self.task == "classification":
            return self._gini(y)
        return float(np.var(y))

    @staticmethod
    def _gini(y):
        """Gini impurity: 1 - Σ pᵢ²"""
        counts = Counter(y)
        n = len(y)
        return 1.0 - sum((c / n) ** 2 for c in counts.values())

    @staticmethod
    def _is_pure(y):
        """True if all labels are identical."""
        return len(np.unique(y)) == 1

    def _leaf_value(self, y):
        """Majority class (classification) or mean (regression)."""
        if self.task == "classification":
            return Counter(y).most_common(1)[0][0]
        return float(np.mean(y))

    # Prediction traversal

    def _traverse(self, x, node):
        """Walk the tree from root to leaf for a single sample."""
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)