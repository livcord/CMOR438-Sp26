import numpy as np
import pytest

from mlpackage.supervised_learning.ensemble import (
    BaggingClassifier,
    AdaBoostClassifier,
)

# Bagging Classifier Tests

def test_bagging_basic_classification():
    """
    Test BaggingClassifier on a simple separable dataset.

    Checks
    - Outputs valid class labels
    - Achieves reasonable accuracy on training data
    """
    X = np.array([[0], [1], [2], [3], [4], [5]])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = BaggingClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    preds = model.predict(X)

    assert set(preds).issubset({0, 1})
    assert np.mean(preds == y) >= 0.8


def test_bagging_output_shape():
    """
    Predictions should match input size.
    """
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, size=20)

    model = BaggingClassifier(n_estimators=5).fit(X, y)
    preds = model.predict(X)

    assert preds.shape == (20,)


def test_bagging_score_consistency():
    """
    score() should match manual accuracy.
    """
    X = np.random.rand(30, 2)
    y = np.random.randint(0, 2, size=30)

    model = BaggingClassifier(n_estimators=5).fit(X, y)

    preds = model.predict(X)
    assert np.isclose(model.score(X, y), np.mean(preds == y))


def test_bagging_oob_score_exists():
    """
    OOB score should exist or be None (depending on sampling overlap).
    """
    X = np.random.rand(50, 4)
    y = np.random.randint(0, 2, size=50)

    model = BaggingClassifier(n_estimators=10, random_state=1)
    model.fit(X, y)

    assert model.oob_score_ is None or (0 <= model.oob_score_ <= 1)


# AdaBoost Tests

def test_adaboost_basic_classification():
    """
    Test AdaBoost on a simple binary dataset.

    Checks
    - Predictions are binary
    - Accuracy is better than random
    """
    X = np.array([[0], [1], [2], [3], [4], [5]])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = AdaBoostClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    preds = model.predict(X)

    assert set(preds).issubset({0, 1})
    assert np.mean(preds == y) >= 0.7


def test_adaboost_decision_function_shape():
    """
    decision_function should return one score per sample.
    """
    X = np.random.rand(25, 2)
    y = np.random.randint(0, 2, size=25)

    model = AdaBoostClassifier(n_estimators=5).fit(X, y)
    scores = model.decision_function(X)

    assert scores.shape == (25,)


def test_adaboost_predict_proba_shape():
    """
    predict_proba should return (n_samples, 2).
    """
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, size=20)

    model = AdaBoostClassifier(n_estimators=5).fit(X, y)
    proba = model.predict_proba(X)

    assert proba.shape == (20, 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_adaboost_score_matches_accuracy():
    """
    score() should match computed accuracy.
    """
    X = np.random.rand(40, 3)
    y = np.random.randint(0, 2, size=40)

    model = AdaBoostClassifier(n_estimators=5).fit(X, y)

    preds = model.predict(X)
    assert np.isclose(model.score(X, y), np.mean(preds == y))

# Edge cases

def test_bagging_invalid_n_estimators():
    with pytest.raises(ValueError):
        BaggingClassifier(n_estimators=0)


def test_bagging_invalid_max_samples():
    with pytest.raises(ValueError):
        BaggingClassifier(max_samples=0)


def test_bagging_invalid_max_features():
    with pytest.raises(ValueError):
        BaggingClassifier(max_features=0)


def test_adaboost_invalid_n_estimators():
    with pytest.raises(ValueError):
        AdaBoostClassifier(n_estimators=0)


def test_adaboost_invalid_learning_rate():
    with pytest.raises(ValueError):
        AdaBoostClassifier(learning_rate=0)


def test_predict_before_fit_bagging():
    model = BaggingClassifier()
    X = np.random.rand(5, 2)

    with pytest.raises(AttributeError):
        model.predict(X)


def test_predict_before_fit_adaboost():
    model = AdaBoostClassifier()
    X = np.random.rand(5, 2)

    with pytest.raises(AttributeError):
        model.predict(X)


def test_adaboost_multiclass_error():
    X = np.random.rand(10, 2)
    y = np.array([0, 1, 2, 1, 0, 2, 1, 0, 1, 2])

    model = AdaBoostClassifier()

    with pytest.raises(ValueError):
        model.fit(X, y)