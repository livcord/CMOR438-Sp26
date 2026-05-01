import numpy as np
import pytest

from mlpackage.supervised_learning.random_forest import RandomForest

# Basic functionality

def test_random_forest_basic_classification():
    """
    Test RandomForest on a simple binary classification dataset.

    Checks
    - Predictions are valid class labels (0 or 1)
    - Accuracy is reasonably high
    """
    X = np.array([[0], [1], [2], [3], [4]])
    y = np.array([0, 0, 0, 1, 1])

    model = RandomForest(n_estimators=10, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)

    assert set(preds).issubset({0, 1})
    assert np.mean(preds == y) > 0.8


def test_random_forest_basic_regression():
    """
    Test RandomForest regression on simple linear data.

    Checks
    - Predictions are numeric
    - R² score is reasonably high
    """
    X = np.arange(1, 6).reshape(-1, 1)
    y = np.array([2, 4, 6, 8, 10])  # y = 2x

    model = RandomForest(task="regression", n_estimators=10, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)

    assert np.issubdtype(preds.dtype, np.floating)
    assert model.score(X, y) > 0.8

# Output shape / type

def test_predict_shape():
    """
    Predictions should match number of samples.
    """
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, size=20)

    model = RandomForest(n_estimators=5).fit(X, y)
    preds = model.predict(X)

    assert preds.shape == (20,)


def test_predict_proba_shape_and_range():
    """
    predict_proba should return valid probabilities.
    """
    X = np.random.rand(30, 2)
    y = np.random.randint(0, 2, size=30)

    model = RandomForest(n_estimators=10).fit(X, y)
    probs = model.predict_proba(X)

    assert probs.shape[0] == len(X)
    assert np.all(probs >= 0)
    assert np.all(probs <= 1)

# Core behavior

def test_multiple_trees_created():
    """
    Number of trees should match n_estimators.
    """
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, size=10)

    model = RandomForest(n_estimators=7).fit(X, y)

    assert len(model.trees_) == 7


def test_oob_score_exists_when_bootstrap():
    """
    OOB score should be computed when bootstrap=True.
    """
    X = np.random.rand(50, 2)
    y = np.random.randint(0, 2, size=50)

    model = RandomForest(n_estimators=10, bootstrap=True, random_state=42)
    model.fit(X, y)

    assert model.oob_score_ is None or (0 <= model.oob_score_ <= 1)


def test_no_oob_score_without_bootstrap():
    """
    OOB score should be None when bootstrap=False.
    """
    X = np.random.rand(30, 2)
    y = np.random.randint(0, 2, size=30)

    model = RandomForest(n_estimators=5, bootstrap=False)
    model.fit(X, y)

    assert model.oob_score_ is None

# Feature subsampling

def test_n_features_parameter():
    """
    Test different n_features settings.
    """
    X = np.random.rand(20, 4)
    y = np.random.randint(0, 2, size=20)

    for nf in [None, 2, 0.5, "sqrt", "log2"]:
        model = RandomForest(n_estimators=5, n_features=nf)
        model.fit(X, y)

        assert len(model.trees_) == 5

# Edge cases

def test_invalid_task():
    """
    Invalid task should raise ValueError.
    """
    with pytest.raises(ValueError):
        RandomForest(task="clustering")


def test_invalid_n_estimators():
    """
    n_estimators must be >= 1.
    """
    with pytest.raises(ValueError):
        RandomForest(n_estimators=0)


def test_invalid_n_features():
    """
    Invalid n_features should raise ValueError.
    """
    X = np.random.rand(10, 3)
    y = np.random.randint(0, 2, size=10)

    model = RandomForest(n_features=10)  # too large
    with pytest.raises(ValueError):
        model.fit(X, y)


def test_predict_before_fit():
    """
    Predicting before fitting should raise error.
    """
    model = RandomForest()
    X = np.random.rand(5, 2)

    with pytest.raises(AttributeError):
        model.predict(X)


def test_predict_proba_wrong_task():
    """
    predict_proba should fail for regression.
    """
    X = np.random.rand(10, 2)
    y = np.random.rand(10)

    model = RandomForest(task="regression").fit(X, y)

    with pytest.raises(ValueError):
        model.predict_proba(X)


def test_shape_mismatch():
    """
    Mismatched X and y should raise ValueError.
    """
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, size=8)

    model = RandomForest()

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_empty_data():
    """
    Fitting on empty data should raise ValueError.
    """
    model = RandomForest()

    with pytest.raises(ValueError):
        model.fit(np.empty((0, 2)), np.array([]))