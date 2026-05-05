import numpy as np

class DBSCAN:
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

    Parameters
    eps : float, default=0.5
        The maximum distance between two samples for them to be considered
        as neighbors.

    min_samples : int, default=5
        The number of samples in a neighborhood for a point to be considered
        as a core point.

    Attributes
    labels_ : np.ndarray of shape (n_samples,)
        Cluster labels for each point. Noisy samples are given the label -1.
    """

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def _region_query(self, X, point_idx):
        """
        Find all points within `eps` distance of a given point.

        Parameters
        X : np.ndarray of shape (n_samples, n_features)
            Dataset.

        point_idx : int
            Index of the point to query.

        Returns
        np.ndarray
            Indices of neighboring points within `eps` distance.
        """
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, labels, point_idx, neighbors, cluster_id):
        """
        Expand a new cluster by recursively adding density-reachable points.

        Parameters
        X : np.ndarray of shape (n_samples, n_features)
            Dataset.

        labels : np.ndarray
            Array tracking cluster assignments for each point.

        point_idx : int
            Index of the starting core point.

        neighbors : np.ndarray
            Indices of neighboring points.

        cluster_id : int
            The current cluster label being assigned.
        """
        labels[point_idx] = cluster_id

        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id

            elif labels[neighbor_idx] == 0:
                labels[neighbor_idx] = cluster_id
                new_neighbors = self._region_query(X, neighbor_idx)

                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.concatenate((neighbors, new_neighbors))

            i += 1

    def fit(self, X):
        """
        Perform DBSCAN clustering on the dataset.

        Parameters
        X : np.ndarray of shape (n_samples, n_features)
            Input data to cluster.

        Returns
        None
            The result is stored in the `labels_` attribute.
        """
        n_samples = X.shape[0]
        labels = np.zeros(n_samples)  # 0 = unvisited, -1 = noise

        cluster_id = 0

        for point_idx in range(n_samples):
            if labels[point_idx] != 0:
                continue

            neighbors = self._region_query(X, point_idx)

            if len(neighbors) < self.min_samples:
                labels[point_idx] = -1
            else:
                cluster_id += 1
                self._expand_cluster(X, labels, point_idx, neighbors, cluster_id)

        self.labels_ = labels.astype(int)