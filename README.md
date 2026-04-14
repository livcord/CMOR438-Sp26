# Machine Learning Algorithms Package

## Overview

This project implements a collection of machine learning algorithms from scratch as part of the final project for CMOR 438: Data Science & Machine Learning. It includes both **supervised** and **unsupervised** learning methods, along with supporting utilities for preprocessing, evaluation, and experimentation.

The repository is organized into:

* A reusable Python package (`mlpackage`)
* Jupyter notebooks demonstrating usage
* Unit tests using pytest

---

## Repository Structure

```
.
├── examples
│   ├── supervised learning
│   │   ├── Perceptron
│   │   ├── Linear Regression
│   │   ├── Logistic Regression
│   │   ├── Multilayer Perceptron
│   │   ├── K Nearest Neighbors
│   │   ├── Decision Trees
│   │   ├── Random Forests
│   │   └── Ensemble Methods
│   └── unsupervised learning
│       ├── Principal Component Analysis
│       ├── K-means
│       └── DSCAN
│
├── src
│   └── mlpackage
│       ├── __init__.py
│       ├── metrics.py
│       ├── preprocess.py
│       ├── supervised_learning
│       │   ├── perceptron.py
│       │   ├── linear_regression.py
│       │   ├── logistic_regression.py
│       │   ├── mlp.py
│       │   ├── knn.py
│       │   ├── decision_tree.py
│       │   ├── random_forest.py
│       │   └── ensemble.py
│       └── unsupervised_learning
│           ├── pca.py
│           ├── kmeans.py
│           └── dbscan.py
│
├── tests
│   ├── __init__.py
│   ├── test_metrics.py
│   ├── test_preprocess.py
│   ├── test_perceptron.py
│   ├── test_linear_regression.py
│   ├── test_logistic_regression.py
│   ├── test_mlp.py
│   ├── test_knn.py
│   ├── test_decision_tree.py
│   ├── test_random_forest.py
│   ├── test_ensemble.py
│   ├── test_pca.py
│   ├── test_kmeans.py
│   └── test_dbscan.py
│
├── README.md
├── requirements.txt
├── pyproject.toml
├── pytest.ini
└── setup.py
```

---

## Implemented Algorithms

### Supervised Learning

* Perceptron
* Linear Regression
* Logistic Regression
* Multilayer Perceptron (MLP)
* K-Nearest Neighbors (KNN)
* Decision Trees
* Random Forests
* Ensemble Methods

### Unsupervised Learning

* Principal Component Analysis (PCA)
* K-Means Clustering
* DBSCAN

---

## Installation

Clone the repository:

```bash
git clone https://github.com/livcord/CMOR438-Sp26.git cd CMOR438-Sp26 pip install -r requirements.txt pip install -e .
```

---

## Usage

Example:

```python
from mlpackage.supervised_learning.logistic_regression import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

## Examples

See the `examples/` directory for Jupyter notebooks demonstrating:

* Data exploration
* Preprocessing
* Model training
* Evaluation

---

## Testing

Run all tests with:

```bash
pytest
```

---

## Author

Livia Cordeiro
Rice University
Spring 2026
