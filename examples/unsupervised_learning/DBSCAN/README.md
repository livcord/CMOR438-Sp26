# DBSCAN

## Overview

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) is an unsupervised clustering algorithm that groups together points that are closely packed (connected through dense regions) while marking isolated points as **noise (outliers)**. Unlike K-Means, DBSCAN does not require specifying the number of clusters in advance and can discover clusters of arbitrary shape.

## How It Works

DBSCAN defines clusters through two parameters:
- **ε (epsilon)**: the neighborhood radius, how close two points must be to be considered neighbors
- **min_samples**: the minimum number of points required within ε to form a dense region

Each point is classified as one of three types:

| Point Type | Condition |
|---|---|
| **Core point** | Has ≥ `min_samples` neighbors within radius ε |
| **Border point** | Within ε of a core point but has fewer than `min_samples` neighbors itself |
| **Noise point** | Not within ε of any core point — labeled **−1** |

**Algorithm:**
1. For each unvisited point, find all points within ε (its neighborhood)
2. If the neighborhood has ≥ `min_samples` points, start a new cluster and expand it recursively: any core point reachable from another core point joins the same cluster
3. Border points join the cluster of the core point they neighbor
4. Points that cannot be reached from any core point are labeled noise (−1)

## Key Properties

| Property | Detail |
|---|---|
| Type | Unsupervised — Clustering + Outlier Detection |
| Number of clusters | **Not required** — discovered automatically |
| Cluster shape | Arbitrary (density-connected, not centroid-based) |
| Outlier detection | Built-in — isolated points are labeled −1 |
| Feature scaling | **Required** — ε is a distance threshold |
| Key weakness | Struggles with clusters of very different densities |

## Choosing ε and min_samples

**min_samples**: A rule of thumb is $2 \times d$ where $d$ is the number of features. For 2D data, start with `min_samples=4` or `5`.

**ε**: Use the **k-distance plot**:
1. For each point, compute its distance to its $k$-th nearest neighbor (where $k$ = `min_samples`)
2. Sort these distances and plot them
3. The elbow of this curve is the recommended ε. Below the elbow, points are in dense regions; above, they are in sparse or noise regions

## Dataset

**Synthetic moons + noise** — generated with `sklearn.make_moons` and random noise injection.

| Column | Description |
|---|---|
| `x` | First coordinate |
| `y` | Second coordinate |
| `is_noise` | 1 = injected noise point, 0 = cluster member (for evaluation) |

**Dataset source:** Generated in the first notebook cell via `sklearn.make_moons` and saved to `moons_noise.csv`. The dataset was also uploaded to this subfolder.

## Files

```
DBSCAN/
├── README.md
├── dbscan.ipynb
└── moons_noise.csv
```

## Usage

```python
from mlpackage.unsupervised_learning.dbscan import DBSCAN
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = DBSCAN(eps=0.3, min_samples=5)
labels = model.fit_predict(X_scaled)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = (labels == -1).sum()
print(f"Clusters: {n_clusters}, Noise points: {n_noise}")
```