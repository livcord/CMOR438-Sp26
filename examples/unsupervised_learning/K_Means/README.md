# K-Means Clustering

## Overview

**K-Means** is an unsupervised clustering algorithm that partitions $n$ samples into $K$ clusters, where each sample belongs to the cluster whose **centroid** (mean) is closest. Unlike supervised methods, K-Means works without labels. It discovers structure in the data purely from feature similarity.

## How It Works

K-Means alternates between two steps until convergence:

**Assignment step:** assign each point to the nearest centroid:
$$c_i = \arg\min_k \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$

**Update step:** recompute each centroid as the mean of its assigned points:
$$\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{i \in C_k} \mathbf{x}_i$$

This minimizes the **Within-Cluster Sum of Squares (WCSS)**:
$$\text{WCSS} = \sum_{k=1}^K \sum_{i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$

The algorithm is guaranteed to converge (WCSS decreases monotonically), but may find a local rather than global minimum. Therefore, multiple restarts with different initializations are recommended.

## Choosing K — The Elbow Method

K-Means requires the number of clusters $K$ in advance. The elbow method helps us find $K$:

1. Run K-Means for $K = 1, 2, \ldots, K_{\max}$
2. Plot WCSS vs $K$ — the curve always decreases
3. Pick the $K$ at the **elbow**: the point where the rate of decrease sharply drops

Beyond the elbow, adding clusters gives diminishing compactness improvement.

## Key Properties

| Property | Detail |
|---|---|
| Type | Unsupervised — Clustering |
| K selection | Manual (elbow, silhouette, or domain knowledge) |
| Cluster shape | Assumes **spherical, equal-size** clusters (Voronoi cells) |
| Feature scaling | **Required** — distance-based algorithm |
| Initialization sensitivity | Yes — multiple restarts recommended |
| Limitation | Cannot find non-convex or unequal-density clusters |

## Dataset

**Synthetic Customer Segments** 

| Column | Description |
|---|---|
| `income` | Annual income (synthetic, thousands $) |
| `spending_score` | Spending behavior score (synthetic, 1–100) |
| `true_cluster` | Ground-truth cluster label (for evaluation only) |

**Dataset source:** Generated in the first notebook cell via `sklearn.datasets.make_blobs` and saved to `customer_segments.csv`, which is attached to this subfolder.

## Files

```
K_Means/
├── README.md
├── kmeans.ipynb
└── customer_segments.csv
```

## Usage

```python
from mlpackage.unsupervised_learning.kmeans import KMeans
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(k=5, random_state=42)
model.fit(X_scaled)
labels = model.predict(X_scaled)
print(f"Inertia (WCSS): {model.inertia_:.2f}")
```