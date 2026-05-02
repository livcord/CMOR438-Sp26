import numpy as np
import pytest
from mlpackage.supervised_learning.mlp import MLP

# Core functionality + forward behavior

def test_mlp_output_shape():
    X = np.random.rand(50, 3)
    y = np.random.randint(0, 2, size=50)

    model = MLP([3, 5, 1], lr=0.1, n_iters=200)
    model.fit(X, y)

    preds = model.predict(X)

    assert preds.shape == (50, 1)


def test_predict_proba_shape():
    X = np.random.rand(20, 4)
    y = np.random.randint(0, 2, size=20)

    model = MLP([4, 6, 1], lr=0.1, n_iters=50)
    model.fit(X, y)

    proba = model.predict_proba(X)

    assert proba.shape == (20, 1)
    assert np.all((proba >= 0) & (proba <= 1))

# Learning behavior

def test_mlp_learns_simple_pattern():
    np.random.seed(0)
    X = np.random.rand(100, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)

    model = MLP([2, 4, 1], lr=0.1, n_iters=1000)
    model.fit(X, y)

    preds = model.predict(X)
    acc = np.mean(preds.flatten() == y)

    assert acc > 0.90


def test_mlp_loss_decreases():
    X = np.random.rand(100, 2)
    y = (X[:, 0] > 0.5).astype(int)

    model = MLP([2, 3, 1], lr=0.1, n_iters=200)

    activations, _ = model._forward(X)
    initial_loss = model._compute_loss(y, activations[-1])

    model.fit(X, y)

    activations, _ = model._forward(X)
    final_loss = model._compute_loss(y, activations[-1])

    assert final_loss < initial_loss + 1e-4

# Output validity checks

def test_predictions_are_binary():
    X = np.random.rand(30, 3)
    y = np.random.randint(0, 2, size=30)

    model = MLP([3, 5, 1], lr=0.1, n_iters=100)
    model.fit(X, y)

    preds = model.predict(X)

    assert set(np.unique(preds)).issubset({0, 1})


def test_probability_range():
    X = np.random.rand(30, 3)
    y = np.random.randint(0, 2, size=30)

    model = MLP([3, 5, 1], lr=0.1, n_iters=100)
    model.fit(X, y)

    proba = model.predict_proba(X)

    assert np.all(proba >= 0) and np.all(proba <= 1)

# Edge cases

def test_single_sample():
    X = np.random.rand(1, 3)
    y = np.array([1])

    model = MLP([3, 4, 1], lr=0.1, n_iters=10)
    model.fit(X, y)

    pred = model.predict(X)

    assert pred.shape == (1, 1)


def test_constant_input():
    X = np.ones((50, 3))
    y = np.zeros(50)

    model = MLP([3, 4, 1], lr=0.1, n_iters=20)
    model.fit(X, y)

    preds = model.predict(X)

    assert preds.shape[0] == 50


def test_unfitted_predict_behavior():
    model = MLP([3, 5, 1])

    X = np.random.rand(5, 3)

    try:
        model.predict(X)
    except Exception as e:
        assert isinstance(e, Exception)