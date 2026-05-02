# Logistic Regression

## Overview

**Logistic Regression** is a supervised classification algorithm that models the **probability** that a sample belongs to a class. Despite its name, it is a classification, and not regression, algorithm. It extends linear regression by squashing the output through the **sigmoid function** to produce probabilities in [0, 1].

## How It Works

The model computes a linear score, then maps it to a probability:

$$P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w} \cdot \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w} \cdot \mathbf{x} + b)}}$$

The **sigmoid function** σ smoothly maps any real number to (0, 1), making the output interpretable as a probability.

A sample is classified as class 1 if P(y=1|x) ≥ threshold (default 0.5), otherwise class 0.

## Training: Binary Cross-Entropy Loss

We minimize log-loss (binary cross-entropy), which heavily penalizes confident wrong predictions:

$$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{p}_i) + (1-y_i) \log(1-\hat{p}_i) \right]$$

Gradient descent updates:

$$\mathbf{w} \leftarrow \mathbf{w} - \frac{\eta}{n} \mathbf{X}^T (\hat{\mathbf{p}} - \mathbf{y})$$

## Key Properties

| Property | Detail |
|---|---|
| Type | Supervised — Binary Classification |
| Output | Probability in [0, 1] + hard label |
| Decision boundary | Linear hyperplane |
| Loss function | Binary cross-entropy |
| Key strength | Interpretable coefficients, probability outputs |

## Decision Threshold

The threshold (default 0.5) can be tuned:
- **Lower threshold** → more positive predictions → higher recall, lower precision
- **Higher threshold** → fewer positive predictions → higher precision, lower recall

This is critical in imbalanced datasets (e.g., fraud detection, medical diagnosis).

## Dataset Used

**Breast Cancer Wisconsin** — UCI ML Repository (via sklearn)  
Binary classification: predict whether a tumor is malignant (1) or benign (0) based on 30 features derived from digitized images of fine needle aspirate (FNA) of breast masses.

**Download:** Run the first cell of `logistic_regression.ipynb` — the notebook exports the dataset from sklearn to CSV automatically. The file is also attached to this subfolder.

## Files

```
Logistic_Regression/
├── README.md
├── logistic_regression.ipynb
└── breast_cancer.csv
```

## Usage

```python
from mlpackage.supervised_learning.logistic_regression import LogisticRegression

model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)    # probabilities
preds = model.predict(X_test)          # hard labels
print(f"Accuracy: {model.score(X_test, y_test):.4f}")
```