import numpy as np
from collections import Counter
from .decision_tree import DecisionTree


class RandomForest:
    """
    Random Forest — an ensemble of Decision Trees trained with two sources
    of randomness to reduce variance:

      1. **Bootstrap sampling** — each tree is trained on a random sample
         (with replacement) of the training data.
      2. **Feature subsampling** — at every split, each tree only considers
         a random subset of features.

    Aggregation:
      - Classification : majority vote across all trees
      - Regression     : mean prediction across all trees

    Parameters
    n_estimators : int
        Number of trees in the forest. Default is 100.
    task : str
        'classification' or 'regression'. Default is 'classification'.
    max_depth : int or None
        Maximum depth of each tree. Default is None.
    min_samples_split : int
        Minimum samples required to split a node. Default is 2.
    min_samples_leaf : int
        Minimum samples required at a leaf. Default is 1.
    n_features : int, float, str, or None
        Number of features to consider at each split.
        - int   : exact number of features
        - float : fraction of total features  (e.g. 0.5)
        - 'sqrt': int(sqrt(n_features))  ← default for classification
        - 'log2': int(log2(n_features))
        - None  : all features (disables feature subsampling)
        Default is 'sqrt'.
    bootstrap : bool
        Whether to use bootstrap sampling. If False, each tree is trained
        on the full dataset (pasting). Default is True.
    random_state : int or None
        Seed for reproducibility. Default is None.

    Attributes
    trees_ : list of DecisionTree
        The fitted trees.
    n_features_in_ : int
        Number of features seen during fit.
    oob_score_ : float or None
        Out-of-bag accuracy/R² estimate (only computed when bootstrap=True).
    """

    VALID_TASKS = ("classification", "regression")

    def __init__(
        self,
        n_estimators=100,
        task="classification",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_features="sqrt",
        bootstrap=True,
        random_state=None,
    ):
        if task not in self.VALID_TASKS:
            raise ValueError(f"task must be one of {self.VALID_TASKS}.")
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")

        self.n_estimators = n_estimators
        self.task = task
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_features = n_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.trees_ = []
        self.n_features_in_ = None
        self.oob_score_ = None
        self._oob_indices = []   # bootstrap indices per tree (for OOB)


    def fit(self, X, y):
        """
        Grow the forest by fitting n_estimators trees.

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

        rng = np.random.default_rng(self.random_state)
        n_samples, n_features_total = X.shape
        self.n_features_in_ = n_features_total

        # Resolve n_features_per_split
        n_feat_split = self._resolve_n_features(n_features_total)

        self.trees_ = []
        self._oob_indices = []

        for _ in range(self.n_estimators):
            # Bootstrap sample 
            if self.bootstrap:
                indices = rng.integers(0, n_samples, size=n_samples)
            else:
                indices = np.arange(n_samples)

            X_boot = X[indices]
            y_boot = y[indices]
            self._oob_indices.append(indices)

            # Build tree with feature subsampling 
            tree = DecisionTree(
                task=self.task,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                n_features=n_feat_split,
            )
            tree.fit(X_boot, y_boot)
            self.trees_.append(tree)

        # Out-of-bag score
        if self.bootstrap:
            self.oob_score_ = self._compute_oob_score(X, y)

        return self

    def predict(self, X):
        """
        Aggregate predictions from all trees.

        Parameters
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        np.ndarray of shape (n_samples,)
        """
        if not self.trees_:
            raise AttributeError("Model must be fitted before calling predict.")

        X = np.array(X, dtype=float)
        # tree_preds shape: (n_estimators, n_samples)
        tree_preds = np.array([tree.predict(X) for tree in self.trees_])

        if self.task == "classification":
            # Majority vote across rows
            return np.array([
                Counter(tree_preds[:, i]).most_common(1)[0][0]
                for i in range(X.shape[0])
            ])
        else:
            return tree_preds.mean(axis=0)

    def predict_proba(self, X):
        """
        Return class probability estimates (classification only).

        Probabilities are computed as the fraction of trees voting for each
        class.

        Parameters
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        np.ndarray of shape (n_samples, n_classes)
        """
        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification.")
        if not self.trees_:
            raise AttributeError("Model must be fitted before calling predict_proba.")

        X = np.array(X, dtype=float)
        tree_preds = np.array([tree.predict(X) for tree in self.trees_])

        classes = np.unique(tree_preds)
        n_samples = X.shape[0]
        proba = np.zeros((n_samples, len(classes)))

        for i, cls in enumerate(classes):
            proba[:, i] = (tree_preds == cls).mean(axis=0)

        return proba

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

    # Internal helpers

    def _resolve_n_features(self, n_total):
        """Convert the n_features parameter to an integer."""
        nf = self.n_features
        if nf is None:
            return n_total
        if isinstance(nf, int):
            if nf < 1 or nf > n_total:
                raise ValueError(f"n_features={nf} is out of range [1, {n_total}].")
            return nf
        if isinstance(nf, float):
            if not (0 < nf <= 1):
                raise ValueError("n_features as float must be in (0, 1].")
            return max(1, int(nf * n_total))
        if nf == "sqrt":
            return max(1, int(np.sqrt(n_total)))
        if nf == "log2":
            return max(1, int(np.log2(n_total)))
        raise ValueError(
            "n_features must be an int, float, 'sqrt', 'log2', or None."
        )

    def _compute_oob_score(self, X, y):
        """
        Estimate generalization error using out-of-bag samples.

        For each training sample, only trees that did NOT see it during
        training vote/predict on it.
        """
        n_samples = X.shape[0]
        oob_preds = [[] for _ in range(n_samples)]

        for tree, boot_idx in zip(self.trees_, self._oob_indices):
            in_bag = set(boot_idx)
            oob_idx = [i for i in range(n_samples) if i not in in_bag]
            if not oob_idx:
                continue
            preds = tree.predict(X[oob_idx])
            for sample_idx, pred in zip(oob_idx, preds):
                oob_preds[sample_idx].append(pred)

        # Only score samples that appeared in at least one OOB set
        valid = [(i, votes) for i, votes in enumerate(oob_preds) if votes]
        if not valid:
            return None

        indices, votes_list = zip(*valid)
        y_true = y[list(indices)]

        if self.task == "classification":
            y_pred = np.array([Counter(v).most_common(1)[0][0] for v in votes_list])
            return np.mean(y_pred == y_true)
        else:
            y_pred = np.array([np.mean(v) for v in votes_list])
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0