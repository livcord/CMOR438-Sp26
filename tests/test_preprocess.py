
import numpy as np
import pytest

from mlpackage.preprocess import (
    train_test_split,
    StandardScaler,
    MinMaxScaler,
    one_hot_encode,
)

# Train/Test Split

def test_split_sizes():
    """
    Test correct split sizes.
    """
    X = np.random.rand(100, 2)
    y = np.random.rand(100)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    assert len(X_test) == 20
    assert len(X_train) == 80
    assert len(X_train) + len(X_test) == 100


def test_split_reproducibility():
    """
    Test deterministic behavior with random_state.
    """
    X = np.random.rand(50, 2)
    y = np.random.rand(50)

    split1 = train_test_split(X, y, random_state=42)
    split2 = train_test_split(X, y, random_state=42)

    assert np.array_equal(split1[0], split2[0])
    assert np.array_equal(split1[1], split2[1])


def test_split_no_shuffle():
    """
    Test behavior without shuffling.
    """
    X = np.arange(10).reshape(-1, 1)
    y = np.arange(10)

    X_train, X_test, _, _ = train_test_split(X, y, shuffle=False, test_size=0.3)

    assert np.array_equal(X_test.flatten(), np.array([0, 1, 2]))

# Standard Scaler

def test_standard_scaler_mean_std():
    """
    Scaled data should have ~0 mean and unit variance.
    """
    X = np.random.rand(100, 3)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.allclose(np.mean(X_scaled, axis=0), 0, atol=1e-7)
    assert np.allclose(np.std(X_scaled, axis=0), 1, atol=1e-7)


def test_standard_scaler_fit_transform_consistency():
    """
    fit_transform should match fit + transform.
    """
    X = np.random.rand(50, 2)

    scaler1 = StandardScaler()
    scaler2 = StandardScaler()

    X1 = scaler1.fit_transform(X)
    scaler2.fit(X)
    X2 = scaler2.transform(X)

    assert np.allclose(X1, X2)

# Min Max Scaler

def test_minmax_range():
    """
    Output should be in [0, 1].
    """
    X = np.random.rand(100, 2)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.all(X_scaled >= 0)
    assert np.all(X_scaled <= 1)


def test_minmax_fit_transform_consistency():
    """
    fit_transform should match fit + transform.
    """
    X = np.random.rand(30, 2)

    scaler1 = MinMaxScaler()
    scaler2 = MinMaxScaler()

    X1 = scaler1.fit_transform(X)
    scaler2.fit(X)
    X2 = scaler2.transform(X)

    assert np.allclose(X1, X2)

# One-hot encoding

def test_one_hot_shape():
    """
    Output shape should be (n_samples, n_classes).
    """
    y = np.array([0, 1, 2, 1])

    encoded = one_hot_encode(y)
    assert encoded.shape == (4, 3)


def test_one_hot_validity():
    """
    Each row should have exactly one '1'.
    """
    y = np.array([1, 0, 1, 2])

    encoded = one_hot_encode(y)
    assert np.all(encoded.sum(axis=1) == 1)


def test_one_hot_consistency():
    """
    Same labels should map to same encoding.
    """
    y = np.array([2, 2, 1, 1])

    encoded = one_hot_encode(y)

    assert np.array_equal(encoded[0], encoded[1])
    assert np.array_equal(encoded[2], encoded[3])