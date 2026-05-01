import numpy as np
import pytest

from mlpackage.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Regression metrics

def test_mse_zero_for_perfect_predictions():
    """
    MSE should be 0 when predictions match targets exactly.
    """
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])

    assert mean_squared_error(y_true, y_pred) == 0.0


def test_rmse_matches_mse():
    """
    RMSE should be the square root of MSE.
    """
    y_true = np.array([1, 2, 3])
    y_pred = np.array([2, 2, 2])

    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)

    assert np.isclose(rmse, np.sqrt(mse))


def test_mae_basic():
    """
    MAE should compute average absolute difference.
    """
    y_true = np.array([1, 2, 3])
    y_pred = np.array([2, 2, 4])

    assert mean_absolute_error(y_true, y_pred) == pytest.approx(2/3)


def test_r2_perfect():
    """
    R² should be 1 for perfect predictions.
    """
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])

    assert r2_score(y_true, y_pred) == 1.0


def test_r2_worse_than_mean():
    """
    R² can be negative if model is worse than predicting the mean.
    """
    y_true = np.array([1, 2, 3])
    y_pred = np.array([10, 10, 10])

    assert r2_score(y_true, y_pred) < 0

# Classification metrics

def test_accuracy_perfect():
    """
    Accuracy should be 1 for perfect predictions.
    """
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0, 1, 1])

    assert accuracy_score(y_true, y_pred) == 1.0


def test_accuracy_basic():
    """
    Accuracy should reflect correct proportion.
    """
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    assert accuracy_score(y_true, y_pred) == 0.75


def test_precision_basic():
    """
    Precision should be TP / (TP + FP).
    """
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 1, 0])  # TP=2, FP=1

    assert precision_score(y_true, y_pred) == pytest.approx(2/3)


def test_precision_zero_division():
    """
    Precision should return 0 when no positive predictions.
    """
    y_true = np.array([1, 1, 0])
    y_pred = np.array([0, 0, 0])

    assert precision_score(y_true, y_pred) == 0.0


def test_recall_basic():
    """
    Recall should be TP / (TP + FN).
    """
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])  # TP=1, FN=1

    assert recall_score(y_true, y_pred) == 0.5


def test_recall_zero_division():
    """
    Recall should return 0 when no actual positives.
    """
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 1, 1])

    assert recall_score(y_true, y_pred) == 0.0


def test_f1_basic():
    """
    F1 should combine precision and recall correctly.
    """
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 1, 0])

    precision = 2 / 3
    recall = 1.0
    expected_f1 = 2 * (precision * recall) / (precision + recall)

    assert f1_score(y_true, y_pred) == pytest.approx(expected_f1)


def test_f1_zero_case():
    """
    F1 should be 0 when precision and recall are 0.
    """
    y_true = np.array([1, 1, 1])
    y_pred = np.array([0, 0, 0])

    assert f1_score(y_true, y_pred) == 0.0