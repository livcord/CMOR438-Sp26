import numpy as np
from mlpackage.unsupervised_learning.dbscan import DBSCAN


def generate_clusters_with_noise(seed=0):
    np.random.seed(seed)
    X1 = np.random.randn(50, 2)
    X2 = np.random.randn(50, 2) + 5
    noise = np.random.uniform(low=-10, high=10, size=(10, 2))
    return np.vstack((X1, X2, noise))


def test_dbscan_detects_clusters():
    X = generate_clusters_with_noise()

    model = DBSCAN(eps=1.0, min_samples=5)
    model.fit(X)

    labels = model.labels_

    clusters = set(labels)
    clusters.discard(-1)

    assert len(clusters) >= 2


def test_dbscan_noise_points_present():
    X = generate_clusters_with_noise()

    model = DBSCAN(eps=0.5, min_samples=5)
    model.fit(X)

    labels = model.labels_

    assert -1 in labels  # noise exists


def test_dbscan_all_noise():
    X = np.array([[0, 0], [10, 10], [20, 20]])

    model = DBSCAN(eps=1.0, min_samples=2)
    model.fit(X)

    assert np.all(model.labels_ == -1)


def test_dbscan_single_cluster():
    np.random.seed(0)
    X = np.random.randn(100, 2)

    model = DBSCAN(eps=3.0, min_samples=3)
    model.fit(X)

    labels = model.labels_
    clusters = set(labels)
    clusters.discard(-1)

    assert len(clusters) == 1


def test_dbscan_labels_shape():
    X = np.random.rand(25, 3)

    model = DBSCAN(eps=0.5, min_samples=3)
    model.fit(X)

    assert model.labels_.shape == (25,)