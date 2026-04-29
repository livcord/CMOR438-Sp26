import numpy as np
import pytest
from mlpackage.supervised_learning.knn import KNN

# Classification

def test_knn_basic_classification():
    """
    Test simple KNN classification.

    Checks
    - Model correctly classifies clearly separable points.
    """
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])

    model = KNN(k=1)
    model.fit(X, y)

    preds = model.predict(np.array([[0.5], [2.5]]))
    assert np.array_equal(preds, np.array([0, 1]))


def test_knn_predict_shape_classification():
    """
    Test prediction shape for classification.

    Checks
    - Output shape matches number of samples.
    """
    X = np.random.rand(20, 2)
    y = (X[:, 0] > 0.5).astype(int)

    model = KNN(k=3)
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape == (20,)


def test_knn_predictions_are_valid_classes():
    """
    Test prediction labels.

    Checks
    - Predictions belong to training labels.
    """
    X = np.random.rand(30, 2)
    y = np.random.randint(0, 3, size=30)

    model = KNN(k=3)
    model.fit(X, y)

    preds = model.predict(X)
    assert set(preds).issubset(set(y))


def test_knn_k1_memorization():
    """
    Test k=1 behavior.

    Checks
    - Model memorizes training data.
    """
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, size=10)

    model = KNN(k=1)
    model.fit(X, y)

    preds = model.predict(X)
    assert np.array_equal(preds, y)

# Regression

def test_knn_basic_regression():
    """
    Test simple regression behavior.

    Checks
    - Predictions approximate target values.
    """
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0.0, 1.0, 2.0, 3.0])

    model = KNN(k=1, task="regression")
    model.fit(X, y)

    preds = model.predict(X)
    assert np.allclose(preds, y)


def test_knn_regression_output_type():
    """
    Test regression output type.

    Checks
    - Predictions are floating point numbers.
    """
    X = np.random.rand(20, 2)
    y = X[:, 0] * 2.5

    model = KNN(k=3, task="regression")
    model.fit(X, y)

    preds = model.predict(X)
    assert np.issubdtype(preds.dtype, np.floating)


def test_knn_regression_shape():
    """
    Test regression prediction shape.

    Checks
    - Output shape matches input samples.
    """
    X = np.random.rand(15, 3)
    y = np.random.rand(15)

    model = KNN(k=5, task="regression")
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape == (15,)

# Distance metrics

def test_knn_manhattan_metric():
    """
    Test Manhattan distance metric.

    Checks
    - Model runs and produces valid predictions.
    """
    X = np.random.rand(20, 2)
    y = (X[:, 0] > 0.5).astype(int)

    model = KNN(k=3, metric="manhattan")
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape == (20,)


def test_knn_minkowski_metric():
    """
    Test Minkowski distance metric.

    Checks
    - Model runs without error.
    """
    X = np.random.rand(20, 2)
    y = (X[:, 1] > 0.5).astype(int)

    model = KNN(k=3, metric="minkowski", p=3)
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape == (20,)

# Edge cases

def test_knn_invalid_k():
    """
    Test invalid k value.

    Checks
    - Raises ValueError.
    """
    with pytest.raises(ValueError):
        KNN(k=0)


def test_knn_invalid_task():
    """
    Test invalid task type.

    Checks
    - Raises ValueError.
    """
    with pytest.raises(ValueError):
        KNN(task="invalid")


def test_knn_invalid_metric():
    """
    Test invalid metric.

    Checks
    - Raises ValueError.
    """
    with pytest.raises(ValueError):
        KNN(metric="cosine")


def test_knn_predict_before_fit():
    """
    Test predict before fitting.

    Checks
    - Raises AttributeError.
    """
    model = KNN()
    X = np.random.rand(5, 2)

    with pytest.raises(AttributeError):
        model.predict(X)


def test_knn_mismatched_shapes():
    """
    Test mismatched input shapes.

    Checks
    - Raises ValueError.
    """
    model = KNN()
    X = np.random.rand(10, 2)
    y = np.random.rand(8)

    with pytest.raises(ValueError):
        model.fit(X, y)