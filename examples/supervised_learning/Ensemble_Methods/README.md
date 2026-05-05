# Ensemble Methods

## Overview

**Ensemble methods** combine multiple weak or moderate learners into a single stronger model. This notebook implements and compares two foundational ensemble strategies:

| Method | Strategy | How trees are combined |
|---|---|---|
| **Bagging** | Parallel — independent learners | Majority vote (equal weights) |
| **AdaBoost** | Sequential — each learner fixes predecessor's errors | Weighted vote (by learner quality) |

Both methods are built on Decision Trees but use them in fundamentally different ways.

## Bagging (Bootstrap Aggregating)

Bagging trains multiple copies of a base learner on **different bootstrap samples** of the data and averages their predictions. The diversity comes from data variation. All learners are independent and can be trained in parallel.

$$\hat{y} = \text{mode}\bigl(\{h_1(\mathbf{x}), h_2(\mathbf{x}), \ldots, h_T(\mathbf{x})\}\bigr)$$

Bagging reduces **variance** without significantly affecting bias. It is most effective on high-variance learners like deep decision trees.

## AdaBoost (Adaptive Boosting)

AdaBoost builds learners **sequentially**, with each new learner forced to focus on the samples the previous ensemble got wrong. Sample weights are increased for misclassified examples and decreased for correctly classified ones.

**Algorithm:**
1. Initialize uniform sample weights $w_i = 1/n$
2. For each round $t = 1, \ldots, T$:
   - Train weak learner $h_t$ on the weighted data
   - Compute weighted error: $\epsilon_t = \sum_i w_i \cdot \mathbf{1}[h_t(x_i) \neq y_i]$
   - Compute learner weight: $\alpha_t = \frac{1}{2} \ln\!\left(\frac{1-\epsilon_t}{\epsilon_t}\right)$
   - Update and normalize sample weights: $w_i \leftarrow w_i \exp(-\alpha_t y_i h_t(x_i))$
3. Predict: $H(\mathbf{x}) = \text{sign}\!\left(\sum_t \alpha_t h_t(\mathbf{x})\right)$

Boosting reduces **bias** (models increasingly complex patterns) and variance. Misclassified samples receive exponentially more attention, the algorithm is *adaptive*.

## Bagging vs Boosting

| Aspect | Bagging | Boosting |
|---|---|---|
| Learner dependency | Independent (parallel) | Sequential (dependent) |
| Primary benefit | Variance reduction | Bias + variance reduction |
| Sensitivity to noise | Robust | More sensitive (amplifies noisy labels) |
| Tuning complexity | Low | Higher (learning rate, n_estimators) |
| Risk of overfitting | Low | Can overfit if noise is high |

## Dataset

**Banknote Authentication** — UCI Machine Learning Repository.

| Feature | Description |
|---|---|
| `variance` | Variance of wavelet-transformed image |
| `skewness` | Skewness of wavelet-transformed image |
| `curtosis` | Kurtosis of wavelet-transformed image |
| `entropy` | Entropy of image |
| `class` | Target: 0 = genuine, 1 = forged |

**Download:** First notebook cell downloads automatically from UCI. The dataset is also attached to this subfolder.

## Files

```
Ensemble_Methods/
├── README.md
├── ensemble.ipynb
└── banknote_authentication.csv
```

## Usage

```python
from mlpackage.supervised_learning.ensemble import BaggingClassifier, AdaBoostClassifier

# Bagging
bag = BaggingClassifier(n_estimators=50, random_state=42)
bag.fit(X_train, y_train)
print(f"Bagging accuracy:  {bag.score(X_test, y_test):.4f}")

# AdaBoost
ada = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, max_depth=1)
ada.fit(X_train, y_train)
print(f"AdaBoost accuracy: {ada.score(X_test, y_test):.4f}")
```