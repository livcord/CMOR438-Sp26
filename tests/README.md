# Tests

This folder contains unit tests for the machine learning models implemented in this repository. The goal of these tests is to verify correctness, ensure consistent behavior, and catch regressions as the code evolves.

The tests are written using **pytest** and are designed to be simple, fast, and independent of external datasets.

---

## Structure

Each model has a corresponding test file that focuses on its core functionality:

- `test_dbscan.py`
- `test_decision_tree.py`
- `test_ensemble.py`
- `test_kmeans.py`
- `test_knn.py`
- `test_linear_regression.py`
- `test_logistic_regression.py`
- `test_mlp.py`
- `test_pca.py`
- `test_perceptron.py`
- `test_random_forest.py`

---

## Testing Approach

The tests emphasize **behavior over benchmarks**. Instead of relying on real-world datasets or strict performance thresholds, they use small synthetic datasets to validate key properties of each algorithm.

### What is tested

- **Basic functionality**
  - Model can fit and predict without errors  
  - Outputs have correct shapes and types  

- **Correctness on simple data**
  - Models recover known patterns
  - Classification outputs are valid class labels  
  - Regression outputs approximate expected values  

- **Core algorithm behavior**
  - Loss decreases during training (where applicable)  
  - Models overfit simple datasets when expected  
  - Hyperparameters affect behavior correctly  

- **Edge cases**
  - Empty inputs  
  - Mismatched shapes  
  - Invalid parameter values  
  - Predicting before fitting  

---

## Metrics Testing (`test_metrics.py`)

This file validates evaluation metrics used across models.

### Coverage
- Regression metrics (MSE, RMSE, MAE, R²)
- Classification metrics (accuracy, precision, recall, F1)
- Edge cases like division-by-zero
- Perfect prediction behavior

---

## Preprocessing Testing (`test_preprocess.py`)

This file tests data preprocessing utilities, which are critical for preparing inputs before training models.

### What is covered

- **Train/Test Split**
  - Correct split sizes based on `test_size`
  - No data loss (train + test = original size)
  - Reproducibility with `random_state`
  - Proper shuffling behavior  

- **StandardScaler**
  - Transformed data has ~0 mean and unit variance  
  - `fit_transform` matches separate `fit` + `transform`  
  - Handles constant features safely  

- **MinMaxScaler**
  - Output values fall within [0, 1]  
  - Scaling is applied feature-wise  
  - `fit_transform` consistency  

- **One-Hot Encoding**
  - Correct output shape  
  - Exactly one “1” per row  
  - Handles arbitrary class labels  

These tests ensure preprocessing steps are reliable, since all models depend on them.

---

## Design Principles

- **Deterministic** — Fixed seeds or simple data  
- **Lightweight** — No external datasets required  
- **Focused** — One behavior per test  
- **Readable** — Clear intent through docstrings  

---

## Running the Tests

From the root of the project, run:

```bash
pytest
```

To run a specific test file:
```bash
pytest tests/file_name.py
```