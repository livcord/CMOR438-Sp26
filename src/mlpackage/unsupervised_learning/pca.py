import numpy as np

class PCA:
    """
    Principal Component Analysis (PCA).

    A dimensionality reduction technique that projects data onto
    directions of maximum variance using eigen decomposition of
    the covariance matrix.
    """

    def __init__(self, n_components):
        """
        Initialize PCA model.

        Parameters
        n_components : int
            Number of principal components to retain.
        """
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None

    def fit(self, X):
        """
        Fit PCA on the dataset.

        Computes the principal components by:
        1. Centering the data
        2. Computing covariance matrix
        3. Performing eigen decomposition
        4. Selecting top eigenvectors by variance

        Parameters
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        self
        """
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort by eigenvalue (descending order)
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[:, idxs]

        # Store top components
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]

        return self

    def transform(self, X):
        """
        Project data onto principal components.

        Parameters
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        np.ndarray of shape (n_samples, n_components)
            Transformed data in reduced dimensional space.
        """
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        """
        Fit PCA and transform the data in one step.

        Parameters
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        np.ndarray
            Reduced-dimension representation of X.
        """
        self.fit(X)
        return self.transform(X)