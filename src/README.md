# src

This directory contains the **core implementation of the `mlpackage` library**, where all machine learning algorithms and supporting utilities are implemented from scratch using NumPy.

It is designed as a modular, testable, and scikit-learn–style codebase used across the `examples/` notebooks and `tests/` suite.

---

# Purpose

The `src/mlpackage` package is responsible for:

- Implementing supervised and unsupervised learning algorithms  
- Providing shared preprocessing and evaluation utilities  
- Enforcing a consistent API across all models  
- Supporting reproducible experiments and unit testing  

---

# API Design

All models follow a consistent interface:

## Supervised models

```python
fit(X, y)
predict(X)
```

## Unsupervised models
```python
fit(X)
predict(X) #if applicable
transform(X) #for dimensionality reduction or clustering variants
fit_transform(X)
```

---
# Module Overview

## supervised_learning/
Implements models trained on labeled data:

- Linear models (Perceptron, Linear Regression, Logistic Regression)
- K-Nearest Neighbors (KNN)
- Decision Trees, Random Forests, Ensemble methods
- Multilayer Perceptron (MLP)

## unsupervised_learning/
Implements structure discovery methods:

- Principal Component Analysis (PCA)
- K-Means Clustering
- DBSCAN

## preprocess.py

Preprocessing utilities:
- Train/test splitting
- Feature scaling (StandardScaler, MinMaxScaler)
- Categorical encoding

## metrics.py
Evaluation metrics used across models:

- Classification metrics
- Regression error metrics
- Reporting utilities

---

## Testing

All modules are fully testable using `pytest`.

Testing focuses on:

- Input/output shape validation  
- Numerical correctness  
- Edge cases (unfitted models, invalid inputs)  
- Consistency across repeated runs  

All tests in `tests/` validate implementation correctness directly.

---

## Design Principles

- **Modularity:** each algorithm is self-contained  
- **Clarity:** implementations prioritize interpretability over optimization  
- **Consistency:** unified API across all models  
- **Reproducibility:** deterministic behavior where possible  
- **Educational focus:** written to clearly expose ML concepts  

---

## Summary

The `src` directory is the core implementation layer of `mlpackage`, translating machine learning concepts into a structured and reusable Python package used across examples, experiments, and tests.