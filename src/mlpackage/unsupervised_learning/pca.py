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

        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

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
        X = np.array(X)

        # Center data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Eigen decomposition (stable for symmetric matrices)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Keep only top-k components
        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ = eigenvalues[:self.n_components]

        # Variance ratio
        self.explained_variance_ratio_ = eigenvalues / np.sum(eigenvalues)

        return self

    def transform(self, X):
        """
        Project data onto principal components.

        Parameters
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        np.ndarray of shape (n_samples, n_components)
            Transformed data in reduced-dimensional space.
        """
        X = np.array(X)
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)

    def fit_transform(self, X):
        """
        Fit PCA and transform the data in one step.

        Parameters
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        np.ndarray
            Reduced-dimensional representation of X.
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        """
        Reconstruct data from its principal component representation.

        Parameters
        X : np.ndarray of shape (n_samples, n_components)
            Data in reduced PCA space.

        Returns
        np.ndarray of shape (n_samples, n_features)
            Reconstructed approximation of original data.
        """
        X = np.array(X)
        return np.dot(X, self.components_.T) + self.mean_