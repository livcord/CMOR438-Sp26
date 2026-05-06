import numpy as np

class kmeans:
    """
    K-Means clustering algorithm.

    Parameters
    n_clusters : int, default=3
        The number of clusters to form.

    max_iters : int, default=100
        Maximum number of iterations of the algorithm.

    tol : float, default=1e-4
        Tolerance for convergence. If the change in centroids is less than
        this value, the algorithm stops.

    random_state : int, default=42
        Seed for random number generation to ensure reproducibility.

    Attributes
    centroids : np.ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    labels_ : np.ndarray of shape (n_samples,)
        Index of the cluster each sample belongs to.
    """

    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4, random_state=42, k=None):
        if k is not None:
            n_clusters = k
        
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None

    

    def _init_centroids(self, X):
        """
        Initialize centroids by randomly selecting points from the dataset.

        Parameters
        X : np.ndarray of shape (n_samples, n_features)
            Input dataset.

        Returns
        np.ndarray
            Initial centroids.
        """
        np.random.seed(self.random_state)
        idxs = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[idxs]

    def _compute_distances(self, X, centroids):
        """
        Compute the Euclidean distance between each point and each centroid.

        Parameters
        X : np.ndarray of shape (n_samples, n_features)
            Input dataset.

        centroids : np.ndarray of shape (n_clusters, n_features)
            Current cluster centroids.

        Returns
        np.ndarray of shape (n_samples, n_clusters)
            Distance matrix where each entry (i, j) is the distance between
            sample i and centroid j.
        """
        return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

    def _assign_clusters(self, distances):
        """
        Assign each point to the nearest centroid.

        Parameters
        distances : np.ndarray of shape (n_samples, n_clusters)
            Distance matrix.

        Returns
        np.ndarray of shape (n_samples,)
            Cluster index assignment for each sample.
        """
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        """
        Update centroids by computing the mean of assigned points.

        Parameters
        X : np.ndarray of shape (n_samples, n_features)
            Input dataset.

        labels : np.ndarray of shape (n_samples,)
            Current cluster assignments.

        Returns
        np.ndarray
            Updated centroids.


        If a cluster has no assigned points, its centroid is reinitialized
        randomly from the dataset.
        """
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            points = X[labels == k]
            if len(points) > 0:
                centroids[k] = np.mean(points, axis=0)
            else:
                # handle empty cluster
                centroids[k] = X[np.random.randint(0, X.shape[0])]
        return centroids

    def fit(self, X):
        """
        Compute K-Means clustering.

        Parameters
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        None
            Fitted model stores results in `centroids` and `labels_`.
        """
        self.centroids_ = self._init_centroids(X)

        for _ in range(self.max_iters):
            distances = self._compute_distances(X, self.centroids_)
            labels = self._assign_clusters(distances)

            new_centroids = self._update_centroids(X, labels)

            shift = np.linalg.norm(self.centroids_ - new_centroids)
            self.centroids_ = new_centroids

            if shift < self.tol:
                break

        self.labels_ = labels

        self.inertia_ = np.sum(
            (X - self.centroids_[self.labels_]) ** 2
        )

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters
        X : np.ndarray of shape (n_samples, n_features)
            New data.

        Returns
        np.ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        distances = self._compute_distances(X, self.centroids_)
        return self._assign_clusters(distances)