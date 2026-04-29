import numpy as np
import pytest
from mlpackage.supervised_learning.decision_tree import DecisionTree

# Classification

def test_decision_tree_basic_classification():
    """
    Test simple classification.

    Checks
    - Model correctly classifies a simple separable dataset.
    """
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])

    model = DecisionTree(task="classification")
    model.fit(X, y)

    preds = model.predict(X)
    assert np.array_equal(preds, y)


def test_decision_tree_predict_shape():
    """
    Test prediction shape.

    Checks
    - Output shape matches number of samples.
    """
    X = np.random.rand(20, 2)
    y = (X[:, 0] > 0.5).astype(int)

    model = DecisionTree()
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape == (20,)


def test_decision_tree_valid_labels():
    """
    Test prediction labels.

    Checks
    - Predictions belong to training labels.
    """
    X = np.random.rand(30, 2)
    y = np.random.randint(0, 3, size=30)

    model = DecisionTree()
    model.fit(X, y)

    preds = model.predict(X)
    assert set(preds).issubset(set(y))


def test_decision_tree_overfits_without_depth_limit():
    """
    Test overfitting behavior.

    Checks
    - Tree achieves perfect accuracy on training data.
    """
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, size=20)

    model = DecisionTree()
    model.fit(X, y)

    preds = model.predict(X)
    assert np.array_equal(preds, y)


def test_max_depth_effect():
    """
    Test depth constraint.

    Checks
    - Deeper tree performs at least as well as shallow tree on training data.
    """
    X = np.random.rand(40, 2)
    y = (X[:, 0] > 0.5).astype(int)

    shallow = DecisionTree(max_depth=1).fit(X, y)
    deep = DecisionTree(max_depth=5).fit(X, y)

    assert np.mean(deep.predict(X) == y) >= np.mean(shallow.predict(X) == y)

# Regression

def test_decision_tree_basic_regression():
    """
    Test regression behavior.

    Checks
    - Model approximates simple function.
    """
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0.0, 1.0, 2.0, 3.0])

    model = DecisionTree(task="regression")
    model.fit(X, y)

    preds = model.predict(X)
    assert np.allclose(preds, y)


def test_regression_output_type():
    """
    Test regression output type.

    Checks
    - Predictions are floats.
    """
    X = np.random.rand(20, 2)
    y = X[:, 0] * 2.0

    model = DecisionTree(task="regression")
    model.fit(X, y)

    preds = model.predict(X)
    assert np.issubdtype(preds.dtype, np.floating)


def test_regression_shape():
    """
    Test regression prediction shape.

    Checks
    - Output shape matches input samples.
    """
    X = np.random.rand(15, 3)
    y = np.random.rand(15)

    model = DecisionTree(task="regression")
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape == (15,)

# Edge cases

def test_invalid_task():
    """
    Test invalid task.

    Checks
    - Raises ValueError.
    """
    with pytest.raises(ValueError):
        DecisionTree(task="invalid")


def test_invalid_min_samples_split():
    """
    Test invalid min_samples_split.

    Checks
    - Raises ValueError.
    """
    with pytest.raises(ValueError):
        DecisionTree(min_samples_split=1)


def test_fit_with_1d_input():
    """
    Test invalid X shape.

    Checks
    - Raises ValueError for 1D input.
    """
    model = DecisionTree()
    X = np.array([1, 2, 3])
    y = np.array([0, 1, 0])

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_empty_data():
    """
    Test empty dataset.

    Checks
    - Raises ValueError.
    """
    model = DecisionTree()

    with pytest.raises(ValueError):
        model.fit(np.empty((0, 2)), np.array([]))


def test_predict_before_fit():
    """
    Test predict before fitting.

    Checks
    - Raises AttributeError.
    """
    model = DecisionTree()
    X = np.random.rand(5, 2)

    with pytest.raises(AttributeError):
        model.predict(X)


def test_mismatched_shapes():
    """
    Test mismatched shapes.

    Checks
    - Raises ValueError.
    """
    model = DecisionTree()
    X = np.random.rand(10, 2)
    y = np.random.rand(8)

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_single_sample():
    """
    Test single-sample dataset.

    Checks
    - Model can fit and predict without crashing.
    """
    X = np.array([[1.0, 2.0]])
    y = np.array([1])

    model = DecisionTree()
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape == (1,)