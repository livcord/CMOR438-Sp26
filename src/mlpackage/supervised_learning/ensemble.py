import numpy as np
from collections import Counter
from .decision_tree import DecisionTree

# Bagging Classifier

class BaggingClassifier:
    """
    Bootstrap Aggregating (Bagging) Classifier.

    Trains multiple copies of a base estimator on different random subsets
    (bootstrap samples) of the training data and aggregates predictions by
    majority vote. Bagging reduces variance without increasing bias, making
    it especially effective for high-variance models like deep decision trees.

    Parameters
    base_estimator : object or None
        Any estimator with fit(X, y) and predict(X) methods.
        Defaults to DecisionTree(task='classification').
    n_estimators : int
        Number of estimators to train. Default is 10.
    max_samples : float
        Fraction of training samples to draw for each bootstrap. Default 1.0.
    max_features : float
        Fraction of features to use per estimator. Default 1.0.
    random_state : int or None
        Seed for reproducibility.

    Attributes
    estimators_ : list
        The fitted base estimators.
    estimators_features_ : list of np.ndarray
        Feature indices used by each estimator.
    oob_score_ : float or None
        Out-of-bag accuracy estimate.
    """

    def __init__(
        self,
        base_estimator=None,
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        random_state=None,
    ):
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")
        if not (0 < max_samples <= 1.0):
            raise ValueError("max_samples must be in (0, 1].")
        if not (0 < max_features <= 1.0):
            raise ValueError("max_features must be in (0, 1].")

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state

        self.estimators_ = []
        self.estimators_features_ = []
        self._oob_indices = []
        self.oob_score_ = None

    def fit(self, X, y):
        """
        Fit all base estimators on bootstrap samples.

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
        n_samples, n_features = X.shape
        n_draw_samples = max(1, int(self.max_samples * n_samples))
        n_draw_features = max(1, int(self.max_features * n_features))

        self.estimators_ = []
        self.estimators_features_ = []
        self._oob_indices = []

        for _ in range(self.n_estimators):
            # Bootstrap sample rows
            sample_idx = rng.integers(0, n_samples, size=n_draw_samples)
            # Random feature subset
            feat_idx = rng.choice(n_features, size=n_draw_features, replace=False)
            feat_idx = np.sort(feat_idx)

            X_boot = X[np.ix_(sample_idx, feat_idx)]
            y_boot = y[sample_idx]

            # Clone the base estimator
            estimator = self._make_estimator()
            estimator.fit(X_boot, y_boot)

            self.estimators_.append(estimator)
            self.estimators_features_.append(feat_idx)
            self._oob_indices.append(sample_idx)

        self.oob_score_ = self._compute_oob_score(X, y)
        return self

    def predict(self, X):
        """
        Predict by majority vote.

        Parameters
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        np.ndarray of shape (n_samples,)
        """
        if not self.estimators_:
            raise AttributeError("Model must be fitted before calling predict.")

        X = np.array(X, dtype=float)
        # Each estimator predicts on its feature subset
        all_preds = np.array([
            est.predict(X[:, feat_idx])
            for est, feat_idx in zip(self.estimators_, self.estimators_features_)
        ])  # shape: (n_estimators, n_samples)

        return np.array([
            Counter(all_preds[:, i]).most_common(1)[0][0]
            for i in range(X.shape[0])
        ])

    def score(self, X, y):
        """Return classification accuracy."""
        return np.mean(self.predict(X) == np.array(y))

    def _make_estimator(self):
        """Return a fresh copy of the base estimator."""
        if self.base_estimator is None:
            return DecisionTree(task="classification")
        # Simple copy by re-instantiating with same params
        cls = type(self.base_estimator)
        params = {}
        for attr in ["task", "max_depth", "min_samples_split",
                     "min_samples_leaf", "n_features"]:
            if hasattr(self.base_estimator, attr):
                params[attr] = getattr(self.base_estimator, attr)
        return cls(**params)

    def _compute_oob_score(self, X, y):
        """Out-of-bag accuracy estimate."""
        n_samples = X.shape[0]
        oob_votes = [[] for _ in range(n_samples)]

        for est, feat_idx, boot_idx in zip(
            self.estimators_, self.estimators_features_, self._oob_indices
        ):
            in_bag = set(boot_idx)
            oob_idx = [i for i in range(n_samples) if i not in in_bag]
            if not oob_idx:
                continue
            preds = est.predict(X[np.ix_(oob_idx, feat_idx)])
            for sample_i, pred in zip(oob_idx, preds):
                oob_votes[sample_i].append(pred)

        valid = [(i, v) for i, v in enumerate(oob_votes) if v]
        if not valid:
            return None

        indices, votes_list = zip(*valid)
        y_pred = np.array([Counter(v).most_common(1)[0][0] for v in votes_list])
        return np.mean(y_pred == y[list(indices)])


# AdaBoost Classifier

class AdaBoostClassifier:
    """
    Adaptive Boosting (AdaBoost) for binary classification.

    AdaBoost builds an ensemble **sequentially**. Each new weak learner focuses
    more on the samples that the previous ensemble got wrong by increasing their
    sample weights. The final prediction is a weighted vote of all weak learners.

    Algorithm (SAMME / Discrete AdaBoost):
      1. Initialize sample weights uniformly: w_i = 1/n
      2. For t = 1 … T:
         a. Train weak learner h_t on weighted data
         b. Compute weighted error: ε_t = Σ w_i · 1[h_t(x_i) ≠ y_i]
         c. Compute learner weight: α_t = 0.5 · ln((1 - ε_t) / ε_t)
         d. Update sample weights: w_i ← w_i · exp(−α_t · y_i · h_t(x_i))
         e. Normalize weights
      3. Predict: H(x) = sign(Σ α_t · h_t(x))

    Parameters
    n_estimators : int
        Maximum number of weak learners (boosting rounds). Default is 50.
    learning_rate : float
        Shrinks each estimator's contribution. Trades off n_estimators.
        Default is 1.0.
    max_depth : int
        Maximum depth of each decision stump / tree. Default is 1 (stump).
    random_state : int or None
        Seed for reproducibility.

    Attributes
    estimators_ : list of DecisionTree
        The fitted weak learners.
    estimator_weights_ : np.ndarray
        Weight α_t for each estimator.
    estimator_errors_ : list of float
        Weighted training error at each boosting round.
    classes_ : np.ndarray
        Unique class labels (must be binary).
    """

    def __init__(
        self,
        n_estimators=50,
        learning_rate=1.0,
        max_depth=1,
        random_state=None,
    ):
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        self.classes_ = None

    def fit(self, X, y):
        """
        Fit the AdaBoost ensemble.

        Parameters
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,) — binary labels

        Returns
        self
        """
        X = np.array(X, dtype=float)
        y = np.array(y)

        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if X.shape[0] != len(y):
            raise ValueError("X and y must have the same number of samples.")

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(
                "AdaBoostClassifier only supports binary classification. "
                f"Found {len(self.classes_)} classes."
            )

        # Map labels to {-1, +1} internally
        y_signed = np.where(y == self.classes_[1], 1, -1).astype(float)

        n_samples = X.shape[0]
        weights = np.ones(n_samples) / n_samples

        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []

        for _ in range(self.n_estimators):
            # Train weak learner on weighted data 
            # Simulate weighting by weighted bootstrap sampling
            rng = np.random.default_rng(self.random_state)
            sample_idx = rng.choice(
                n_samples, size=n_samples, replace=True, p=weights
            )
            stump = DecisionTree(
                task="classification", max_depth=self.max_depth
            )
            stump.fit(X[sample_idx], y[sample_idx])

            # Compute weighted error
            y_pred = stump.predict(X)
            y_pred_signed = np.where(y_pred == self.classes_[1], 1, -1).astype(float)

            incorrect = (y_pred_signed != y_signed).astype(float)
            epsilon = np.dot(weights, incorrect)

            # If error is 0 or >= 0.5, stop early
            epsilon = np.clip(epsilon, 1e-10, 1 - 1e-10)
            if epsilon >= 0.5:
                break

            # Compute estimator weight
            alpha = self.learning_rate * 0.5 * np.log((1 - epsilon) / epsilon)

            # Update sample weights
            weights *= np.exp(-alpha * y_signed * y_pred_signed)
            weights /= weights.sum()   # normalize

            self.estimators_.append(stump)
            self.estimator_weights_.append(alpha)
            self.estimator_errors_.append(epsilon)

        self.estimator_weights_ = np.array(self.estimator_weights_)
        return self

    def decision_function(self, X):
        """
        Compute the raw boosted score for each sample.

        Returns
        np.ndarray of shape (n_samples,)
            Positive = class_[1], Negative = class_[0].
        """
        if not self.estimators_:
            raise AttributeError("Model must be fitted before calling decision_function.")

        X = np.array(X, dtype=float)
        scores = np.zeros(X.shape[0])

        for stump, alpha in zip(self.estimators_, self.estimator_weights_):
            y_pred = stump.predict(X)
            y_pred_signed = np.where(y_pred == self.classes_[1], 1, -1).astype(float)
            scores += alpha * y_pred_signed

        return scores

    def predict(self, X):
        """
        Predict binary class labels.

        Returns
        np.ndarray of shape (n_samples,)
        """
        scores = self.decision_function(X)
        return np.where(scores >= 0, self.classes_[1], self.classes_[0])

    def predict_proba(self, X):
        """
        Soft probability estimates via sigmoid of the decision function.

        Returns
        np.ndarray of shape (n_samples, 2)
        """
        scores = self.decision_function(X)
        prob_pos = 1 / (1 + np.exp(-2 * scores))
        return np.column_stack([1 - prob_pos, prob_pos])

    def score(self, X, y):
        """Return classification accuracy."""
        return np.mean(self.predict(X) == np.array(y))

    def staged_score(self, X, y):
        """
        Yield accuracy after each boosting round (useful for learning curves).

        Yields
        float : accuracy after adding each estimator
        """
        X = np.array(X, dtype=float)
        y = np.array(y)
        running_scores = np.zeros(X.shape[0])

        for stump, alpha in zip(self.estimators_, self.estimator_weights_):
            y_pred = stump.predict(X)
            y_pred_signed = np.where(y_pred == self.classes_[1], 1, -1).astype(float)
            running_scores += alpha * y_pred_signed
            preds = np.where(running_scores >= 0, self.classes_[1], self.classes_[0])
            yield np.mean(preds == y)