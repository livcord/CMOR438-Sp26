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

    n_clusters_ : int
        Number of clusters found (excluding noise).

    core_sample_indices_ : np.ndarray
        Indices of core samples.
    """

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

        self.labels_ = None
        self.n_clusters_ = None
        self.core_sample_indices_ = None

    def _region_query(self, X, point_idx):
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, labels, point_idx, neighbors, cluster_id, core_samples):
        labels[point_idx] = cluster_id
        core_samples.add(point_idx)

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
                    core_samples.add(neighbor_idx)

            i += 1

    def fit(self, X):
        X = np.array(X)
        n_samples = X.shape[0]

        labels = np.zeros(n_samples, dtype=int)  # 0 = unvisited
        cluster_id = 0
        core_samples = set()

        for point_idx in range(n_samples):
            if labels[point_idx] != 0:
                continue

            neighbors = self._region_query(X, point_idx)

            if len(neighbors) < self.min_samples:
                labels[point_idx] = -1  # noise
            else:
                cluster_id += 1
                self._expand_cluster(X, labels, point_idx, neighbors, cluster_id, core_samples)

        # Convert cluster labels to start from 0
        labels[labels > 0] -= 1

        self.labels_ = labels
        self.n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        self.core_sample_indices_ = np.array(sorted(core_samples))

        return self

    def fit_predict(self, X):
        """
        Perform clustering and return cluster labels.

        Parameters
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        np.ndarray
            Cluster labels for each sample.
        """
        self.fit(X)
        return self.labels_