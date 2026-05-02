# K-Nearest Neighbors (KNN)

## Overview

**K-Nearest Neighbors** is a non-parametric, instance-based learning algorithm. It stores the entire training set and makes predictions by finding the **K most similar training examples** to a new query point and aggregating their labels. It is called a *lazy learner* because no model is built during training; all computation is deferred to prediction time.

## How It Works

To classify a new point **x**:

1. Compute the distance from **x** to every training sample
2. Rank training samples by distance and select the **K** closest
3. Take a **majority vote** of their class labels (classification) or compute the **mean** (regression)

$$\hat{y} = \text{mode}\bigl(\{y_{(1)}, y_{(2)}, \ldots, y_{(K)}\}\bigr)$$

## Distance Metrics

| Metric | Formula | When to use |
|---|---|---|
| Euclidean | $\sqrt{\sum(x_i - z_i)^2}$ | Default; assumes isotropic feature space |
| Manhattan | $\sum|x_i - z_i|$ | Robust to outliers; grid-like spaces |
| Minkowski | $\left(\sum|x_i-z_i|^p\right)^{1/p}$ | General family (Euclidean at p=2) |

## Choosing K — The Bias-Variance Trade-off

| K | Bias | Variance | Effect |
|---|---|---|---|
| K = 1 | Low | High | Memorizes training data; wiggly boundary |
| K = n | High | Low | Predicts constant (the majority class) |
| Optimal K | Balanced | Balanced | Smooth boundary; generalizes well |

A rule of thumb is to start with $K = \sqrt{n_{\text{train}}}$ and tune via cross-validation.

## Dataset

**Palmer Penguins** — 3-species classification from Palmer Station, Antarctica.

| Feature | Description |
|---|---|
| `bill_length_mm` | Length of the bill (mm) |
| `bill_depth_mm` | Depth of the bill (mm) |
| `flipper_length_mm` | Flipper length (mm) |
| `body_mass_g` | Body mass (g) |
| `species` | Target: Adelie, Chinstrap, Gentoo |

**Download:** The first notebook cell downloads the CSV automatically from the seaborn data repository. The dataset is also attached to this subfolder.

## Files

```
K_Nearest_Neighbors/
├── README.md
├── knn.ipynb
└── penguins.csv
```

## Usage

```python
from mlpackage.supervised_learning.knn import KNN
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = KNN(k=5, task='classification', metric='euclidean')
model.fit(X_train_s, y_train)
print(f"Accuracy: {model.score(X_test_s, y_test):.4f}")
```