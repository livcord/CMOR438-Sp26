import numpy as np
import pytest
from mlpackage.supervised_learning.linear_regression import LinearRegression

# Core functionality

def test_linear_regression_perfect_fit():
    """
    Test predictions on perfectly linear data.

    Checks
    - Model learns exact linear relationship.
    """
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])  # y = 2x

    model = LinearRegression()
    model.fit(X, y)

    preds = model.predict(np.array([[5], [6]]))
    assert np.allclose(preds, [10, 12], atol=1e-6)


def test_linear_regression_predict_shape():
    """
    Test that predict returns correct shape.

    Checks
    - Output shape matches (n_samples,).
    """
    X = np.random.rand(10, 2)
    y = np.random.rand(10)

    model = LinearRegression()
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape == (10,)

# Metrics

def test_linear_regression_rmse_zero_on_perfect_data():
    """
    Test RMSE on perfectly linear data.

    Checks
    - RMSE is effectively zero.
    """
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2, 3])  # y = x

    model = LinearRegression()
    model.fit(X, y)

    assert model.rmse(X, y) < 1e-6


def test_linear_regression_r_squared_perfect():
    """
    Test R² score on perfect data.

    Checks
    - R² ≈ 1.0
    """
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])

    model = LinearRegression()
    model.fit(X, y)

    r2 = model.R_squared(X, y)
    assert np.isclose(r2, 1.0, atol=1e-6)


def test_linear_regression_r_squared_noisy():
    """
    Test R² on noisy linear data.

    Checks
    - R² remains reasonably high.
    """
    rng = np.random.default_rng(42)
    X = np.arange(1, 21).reshape(-1, 1)
    y = 3 * X.flatten() + 5 + rng.normal(0, 1, size=20)

    model = LinearRegression()
    model.fit(X, y)

    r2 = model.R_squared(X, y)
    assert 0.8 <= r2 <= 1.0

# Robustness

def test_linear_regression_predictions_are_finite():
    """
    Test that predictions contain no NaN or inf values.
    """
    X = np.random.rand(20, 3)
    y = np.random.rand(20)

    model = LinearRegression()
    model.fit(X, y)

    preds = model.predict(X)
    assert np.all(np.isfinite(preds))

# Edge cases

def test_linear_regression_empty_data():
    """
    Test fitting on empty data.

    Checks
    - Raises ValueError.
    """
    model = LinearRegression()
    X = np.array([]).reshape(0, 2)
    y = np.array([])

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_linear_regression_mismatched_shapes():
    """
    Test mismatched input sizes.

    Checks
    - Raises ValueError.
    """
    model = LinearRegression()
    X = np.ones((10, 2))
    y = np.ones(9)

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_linear_regression_unfitted_predict():
    """
    Test predict before fit.

    Checks
    - Raises AttributeError.
    """
    model = LinearRegression()
    X = np.random.rand(5, 2)

    with pytest.raises(AttributeError):
        model.predict(X)