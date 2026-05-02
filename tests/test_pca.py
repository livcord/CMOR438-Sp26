import numpy as np
import pytest
from mlpackage.unsupervised_learning.pca import PCA

# Basic functionality

def test_pca_output_shape():
    """
    Ensure PCA reduces dimensionality correctly.
    """
    X = np.random.rand(100, 5)
    pca = PCA(n_components=2)

    X_transformed = pca.fit_transform(X)

    assert X_transformed.shape == (100, 2)


def test_pca_variance_sorted():
    """
    Ensure explained variance is sorted in descending order.
    """
    X = np.random.rand(100, 5)
    pca = PCA(n_components=3)
    pca.fit(X)

    assert np.all(np.diff(pca.explained_variance) <= 0)


# Correctness & consistency 

def test_pca_fit_transform_equivalence():
    """
    fit_transform should match fit + transform.
    """
    X = np.random.rand(80, 6)

    pca1 = PCA(n_components=3)
    pca2 = PCA(n_components=3)

    result1 = pca1.fit_transform(X)
    pca2.fit(X)
    result2 = pca2.transform(X)

    assert np.allclose(result1, result2)


def test_pca_transform_consistency():
    """
    Transform should be consistent after fitting.
    """
    X = np.random.rand(50, 4)

    pca = PCA(n_components=2)
    pca.fit(X)

    first = pca.transform(X)
    second = pca.transform(X)

    assert np.allclose(first, second)


# Mathematical properties 

def test_pca_centering_property():
    """
    PCA should center data before projection (mean ≈ 0 in PCA space).
    """
    X = np.random.rand(100, 5)

    pca = PCA(n_components=3)
    X_transformed = pca.fit_transform(X)

    assert np.allclose(np.mean(X_transformed, axis=0), 0, atol=1e-6)


def test_pca_components_orthogonality():
    """
    Principal components should be orthogonal.
    """
    X = np.random.rand(60, 5)
    pca = PCA(n_components=3)
    pca.fit(X)

    components = pca.components

    # Dot product between different components should be ~0
    gram_matrix = components.T @ components

    identity = np.eye(components.shape[1])

    assert np.allclose(gram_matrix, identity, atol=1e-6)


# Edge cases

def test_pca_full_dimension():
    """
    PCA with all components should preserve shape.
    """
    X = np.random.rand(40, 4)
    pca = PCA(n_components=4)

    X_transformed = pca.fit_transform(X)

    assert X_transformed.shape == X.shape


def test_pca_single_component():
    """
    PCA should work with n_components = 1.
    """
    X = np.random.rand(30, 5)
    pca = PCA(n_components=1)

    X_transformed = pca.fit_transform(X)

    assert X_transformed.shape == (30, 1)


def test_pca_deterministic_behavior():
    """
    PCA should produce consistent results across runs.
    """
    X = np.random.rand(50, 4)

    pca1 = PCA(n_components=2)
    pca2 = PCA(n_components=2)

    out1 = pca1.fit_transform(X)
    out2 = pca2.fit_transform(X)

    assert np.allclose(out1, out2)


# Stability checks

def test_pca_variance_non_negative():
    """
    Explained variance should always be non-negative.
    """
    X = np.random.rand(100, 5)
    pca = PCA(n_components=3)
    pca.fit(X)

    assert np.all(pca.explained_variance >= 0)