import numpy as np
from mlpackage.unsupervised_learning.kmeans import kmeans


def generate_two_clusters(seed=0):
    np.random.seed(seed)
    X1 = np.random.randn(50, 2) + np.array([0, 0])
    X2 = np.random.randn(50, 2) + np.array([5, 5])
    return np.vstack((X1, X2))


def test_kmeans_finds_correct_number_of_clusters():
    X = generate_two_clusters()

    model = kmeans(n_clusters=2)
    model.fit(X)

    labels = model.labels_
    assert len(np.unique(labels)) == 2


def test_kmeans_centroids_shape():
    X = np.random.rand(30, 4)

    model = kmeans(n_clusters=3)
    model.fit(X)

    assert model.centroids_.shape == (3, 4)


def test_kmeans_predict_shape():
    X = np.random.rand(20, 3)

    model = kmeans(n_clusters=3)
    model.fit(X)

    preds = model.predict(X)
    assert preds.shape == (20,)


def test_kmeans_predict_consistency():
    X = generate_two_clusters()

    model = kmeans(n_clusters=2)
    model.fit(X)

    preds1 = model.predict(X)
    preds2 = model.predict(X)

    assert np.array_equal(preds1, preds2)


def test_kmeans_cluster_separation():
    X = generate_two_clusters()

    model = kmeans(n_clusters=2)
    model.fit(X)

    labels = model.labels_

    # first 50 points vs last 50 points should mostly differ
    first_cluster = labels[:50]
    second_cluster = labels[50:]

    # majority label in each half should differ
    assert np.bincount(first_cluster).argmax() != np.bincount(second_cluster).argmax()


def test_kmeans_single_cluster():
    X = np.random.rand(50, 2)

    model = kmeans(n_clusters=1)
    model.fit(X)

    assert np.all(model.labels_ == 0)