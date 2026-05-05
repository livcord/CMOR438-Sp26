# Principal Component Analysis (PCA)

## Overview

**Principal Component Analysis** is an unsupervised dimensionality reduction technique. It finds the directions (principal components) along which the data varies the most, and projects the data onto a lower-dimensional subspace defined by those directions. PCA is not a classifier, it transforms the feature space to make structure easier to visualize and to remove redundant, low-information dimensions.

## How It Works

Given a centered data matrix $\mathbf{X}$ (zero mean per feature):

1. **Compute the covariance matrix**: $\mathbf{C} = \frac{1}{n-1}\mathbf{X}^T\mathbf{X}$
2. **Eigendecomposition**: find eigenvectors $\mathbf{v}_k$ and eigenvalues $\lambda_k$ of $\mathbf{C}$
3. **Sort by eigenvalue** (descending) — the eigenvector with the largest eigenvalue is the direction of maximum variance
4. **Project**: $\mathbf{Z} = \mathbf{X}\mathbf{V}_k$, where $\mathbf{V}_k$ contains the top $k$ eigenvectors

Each eigenvector is a **principal component**, a new axis in feature space. The corresponding eigenvalue tells us how much variance is explained by that component.

## Explained Variance Ratio

$$\text{EVR}_k = \frac{\lambda_k}{\sum_j \lambda_j}$$

The cumulative EVR shows how many components are needed to capture a given fraction (e.g., 95%) of the total variance in the data.

## Key Properties

| Property | Detail |
|---|---|
| Type | Unsupervised — Dimensionality Reduction |
| Input | High-dimensional feature matrix |
| Output | Lower-dimensional projection |
| Feature scaling | **Required** — otherwise high-variance features dominate |
| Assumption | Linear structure in the data |
| Information loss | Controlled by number of components retained |

## Dataset

**Wine** (sklearn) — 178 wine samples described by 13 chemical measurements.

| Feature | Description |
|---|---|
| `alcohol` | Alcohol content |
| `malic_acid` | Malic acid |
| `ash` | Ash content |
| `alcalinity_of_ash` | Alcalinity of ash |
| `magnesium` | Magnesium |
| `total_phenols` | Total phenols |
| `flavanoids` | Flavanoids |
| `nonflavanoid_phenols` | Non-flavanoid phenols |
| `proanthocyanins` | Proanthocyanins |
| `color_intensity` | Color intensity |
| `hue` | Hue |
| `od280_od315_of_diluted_wines` | OD280/OD315 |
| `proline` | Proline |
| `target` | Wine cultivar: 0, 1, or 2 |

**Download:** First notebook cell exports automatically from sklearn. The dataset is also attached to this subfolder.

## Files

```
PCA/
├── README.md
├── pca.ipynb
└── wine.csv
```

## Usage

```python
from mlpackage.unsupervised_learning.pca import PCA

pca = PCA(n_components=2)
pca.fit(X_train_s)
X_reduced = pca.transform(X_train_s)   # shape: (n_samples, 2)

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
```