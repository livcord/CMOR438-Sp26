# Decision Trees

## Overview

A **Decision Tree** is a hierarchical model that makes predictions by learning a sequence of simple yes/no questions about the features. Starting at the root, each internal node tests a feature against a threshold, and each leaf holds a final prediction. The resulting model is fully **interpretable**; you can trace the exact chain of conditions that led to any prediction.

## How It Works

The tree is built by recursively finding the **best binary split** at each node:

1. For each feature and each unique threshold value, compute the **information gain**, i.e. the reduction in impurity after the split
2. Choose the feature–threshold pair that maximizes gain
3. Recursively split the left (≤ threshold) and right (> threshold) subsets
4. Stop when a stopping condition is met (pure node, max depth reached, too few samples)

## Splitting Criterion — Gini Impurity

For classification, we measure node impurity using **Gini**:

$$\text{Gini}(S) = 1 - \sum_{k} p_k^2$$

A pure node (all one class) has Gini = 0. A maximally mixed node has Gini = 0.5. We pick the split that maximally reduces weighted Gini across the two child nodes.

## Variance Reduction (Regression)

For regression trees, "impurity" is replaced by **variance**. We pick splits that reduce the variance of target values within each child node.

## Key Hyperparameters

| Parameter | Effect |
|---|---|
| `max_depth` | Primary regularizer — limits depth, prevents overfitting |
| `min_samples_split` | Node must have at least this many samples to be split |
| `min_samples_leaf` | Each leaf must contain at least this many samples |

## Dataset

**Titanic Survival** — the most famous binary classification dataset in machine learning.

| Feature | Description |
|---|---|
| `pclass` | Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| `sex` | Sex (0 = male, 1 = female after encoding) |
| `age` | Age in years |
| `sibsp` | Number of siblings/spouses aboard |
| `parch` | Number of parents/children aboard |
| `fare` | Passenger fare |
| `survived` | Target: 0 = did not survive, 1 = survived |

**Download:** First notebook cell downloads automatically from the seaborn data repository. The dataset is also attached to this subfolder.

## Files

```
Decision_Trees/
├── README.md
├── decision_tree.ipynb
└── titanic.csv
```

## Usage

```python
from mlpackage.supervised_learning.decision_tree import DecisionTree

# No feature scaling needed — trees are scale-invariant
model = DecisionTree(task='classification', max_depth=4)
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.4f}")
```