import numpy as np
import pytest
from mlpackage.supervised_learning.perceptron import Perceptron

# Core functionality

def test_perceptron_fits_linearly_separable():
    """
    Test Perceptron on perfectly linearly separable data.

    Dataset
    - AND gate

    Checks
    - Model perfectly classifies training data.
    """
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 0, 1])

    model = Perceptron(learning_rate=0.1, n_iterations=100)
    model.fit(X, y)

    preds = model.predict(X)
    assert np.array_equal(preds, y)


def test_perceptron_learns_simple_rule():
    """
    Test that Perceptron learns a simple threshold rule.

    Dataset
    - Label = 1 if first feature > 0.5

    Checks
    - Accuracy is high on training data.
    """
    X = np.random.rand(50, 3)
    y = (X[:, 0] > 0.5).astype(int)

    model = Perceptron()
    model.fit(X, y)

    preds = model.predict(X)
    assert np.mean(preds == y) > 0.9


def test_perceptron_predict_shape():
    """
    Test prediction output shape.

    Checks
    - Output shape matches (n_samples,).
    """
    X = np.random.rand(50, 3)
    y = (X[:, 0] > 0.5).astype(int)

    model = Perceptron()
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape == (50,)


def test_perceptron_outputs_binary_labels():
    """
    Test that predictions are valid binary labels.

    Checks
    - Predictions are only 0 or 1.
    """
    X = np.random.rand(30, 2)
    y = (X[:, 1] > 0.5).astype(int)

    model = Perceptron()
    model.fit(X, y)

    preds = model.predict(X)
    assert set(preds).issubset({0, 1})

# Robustness

def test_perceptron_predictions_are_finite():
    """
    Test that predictions contain no NaN or inf values.
    """
    X = np.random.rand(20, 2)
    y = (X[:, 0] > 0.5).astype(int)

    model = Perceptron()
    model.fit(X, y)

    preds = model.predict(X)
    assert np.all(np.isfinite(preds))

# Edge cases

def test_perceptron_unfitted_predict():
    """
    Test prediction before fitting.

    Checks
    - Raises TypeError.
    """
    model = Perceptron()
    X = np.random.rand(5, 2)

    with pytest.raises(TypeError):
        model.predict(X)


def test_perceptron_mismatched_shapes():
    """
    Test mismatched input sizes.

    Checks
    - Raises ValueError.
    """
    model = Perceptron()
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, size=8)

    with pytest.raises(ValueError):
        model.fit(X, y)