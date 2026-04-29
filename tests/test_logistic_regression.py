import numpy as np
import pytest
from mlpackage.supervised_learning.logistic_regression import LogisticRegression

# Core functionality

def test_logistic_regression_basic():
    """
    Test LogisticRegression on a simple linearly separable dataset.

    Checks
    - Predictions are binary.
    - Accuracy is reasonably high.
    """
    X = np.array([[0], [1], [2], [3], [4]])
    y = np.array([0, 0, 0, 1, 1])

    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)

    preds = model.predict(X)

    assert set(preds).issubset({0, 1})
    assert np.mean(preds == y) > 0.8


def test_logistic_regression_fit_predict_shape():
    """
    Test basic fit and predict behavior.

    Checks
    - Prediction shape matches labels.
    - Predictions are binary.
    """
    X = np.array([[0], [1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1, 1])

    model = LogisticRegression()
    model.fit(X, y)

    preds = model.predict(X)

    assert preds.shape == y.shape
    assert set(preds).issubset({0, 1})


def test_logistic_regression_learns_simple_rule():
    """
    Test learning on a simple synthetic rule.

    Checks
    - Model captures a threshold-based pattern.
    """
    X = np.random.rand(50, 2)
    y = (X[:, 0] > 0.5).astype(int)

    model = LogisticRegression(learning_rate=0.1, n_iterations=500)
    model.fit(X, y)

    preds = model.predict(X)

    assert np.mean(preds == y) > 0.85

# Probability outputs

def test_logistic_regression_probability_output():
    """
    Test probability outputs.

    Checks
    - Probabilities are in [0, 1].
    """
    X = np.array([[1], [2]])
    y = np.array([0, 1])

    model = LogisticRegression()
    model.fit(X, y)

    probs = model.predict_proba(X)

    assert np.all((probs >= 0) & (probs <= 1))

def test_logistic_regression_probability_shape():
    """
    Test probability output shape.

    Checks
    - Output shape is (n_samples,).
    """
    X = np.random.rand(10, 3)
    y = (X[:, 1] > 0.5).astype(int)

    model = LogisticRegression()
    model.fit(X, y)

    probs = model.predict_proba(X)

    assert probs.shape == (10,)

# Training behavior

def test_logistic_regression_loss_decreases():
    """
    Test training convergence.

    Checks
    - Loss decreases over iterations.
    """
    X = np.random.rand(50, 2)
    y = (X[:, 0] > 0.5).astype(int)

    model = LogisticRegression(learning_rate=0.1, n_iterations=200)
    model.fit(X, y)

    assert model.loss_history[0] > model.loss_history[-1]


def test_logistic_regression_predictions_are_finite():
    """
    Test numerical stability.

    Checks
    - Predictions contain no NaN or inf values.
    """
    X = np.random.rand(30, 2)
    y = (X[:, 0] > 0.5).astype(int)

    model = LogisticRegression()
    model.fit(X, y)

    preds = model.predict(X)

    assert np.all(np.isfinite(preds))

# Behavior checks

def test_logistic_regression_threshold_effect():
    """
    Test threshold impact.

    Checks
    - Lower threshold increases positive predictions.
    """
    X = np.random.rand(100, 2)
    y = (X[:, 0] > 0.5).astype(int)

    model = LogisticRegression()
    model.fit(X, y)

    preds_default = model.predict(X)

    model.threshold = 0.3
    preds_lower = model.predict(X)

    assert preds_lower.sum() >= preds_default.sum()

# Edge cases

def test_logistic_regression_invalid_threshold():
    """
    Test invalid threshold.

    Checks
    - Raises ValueError.
    """
    with pytest.raises(ValueError):
        LogisticRegression(threshold=1.5)


def test_logistic_regression_invalid_shape():
    """
    Test mismatched shapes.

    Checks
    - Raises ValueError.
    """
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, size=9)

    model = LogisticRegression()

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_logistic_regression_unfitted_predict():
    """
    Test predict before fit.

    Checks
    - Raises TypeError.
    """
    model = LogisticRegression()
    X = np.random.rand(5, 2)

    with pytest.raises(TypeError):
        model.predict(X)


def test_logistic_regression_non_binary_labels():
    """
    Test non-binary targets.

    Checks
    - Raises ValueError.
    """
    X = np.random.rand(10, 2)
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

    model = LogisticRegression()

    with pytest.raises(ValueError):
        model.fit(X, y)