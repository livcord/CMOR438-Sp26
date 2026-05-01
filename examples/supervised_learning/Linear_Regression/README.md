# Linear Regression

## Overview

**Linear Regression** models the relationship between one or more input features and a continuous target variable by fitting a linear equation to the data. It is one of the most widely used and interpretable algorithms in statistics and machine learning.

## How It Works

The model predicts a continuous output as a weighted sum of inputs:

$$\hat{y} = \mathbf{w} \cdot \mathbf{x} + b = w_1 x_1 + w_2 x_2 + \ldots + w_n x_n + b$$

Training minimizes **Mean Squared Error (MSE)**:

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

This implementation supports two fitting methods:

### Method 1 — Ordinary Least Squares (OLS)
Solves analytically via the **Normal Equation**:

$$\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

Exact solution, but expensive for large feature sets (O(d³) inversion).

### Method 2 — Gradient Descent (GD)
Iteratively updates parameters in the direction of steepest descent:

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \frac{\partial \text{MSE}}{\partial \mathbf{w}} = \mathbf{w} - \frac{\eta}{n} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y})$$

Scalable to large datasets; requires tuning of learning rate and iterations.

## Key Properties

| Property | Detail |
|---|---|
| Type | Supervised — Regression |
| Output | Continuous real-valued prediction |
| Assumption | Linear relationship between features and target |
| Key metric | R² (coefficient of determination), RMSE |

## Evaluation Metrics

- **MSE / RMSE** — penalizes large errors; in the same units as the target (RMSE)
- **R²** — fraction of variance explained; R²=1 is a perfect fit, R²=0 means no better than predicting the mean

## Dataset Used

**Advertising** — classic marketing dataset  
Predict product `Sales` (in thousands of units) from advertising spend on TV, Radio, and Newspaper.

| Column | Description |
|---|---|
| `TV` | Advertising spend on TV (thousands $) |
| `radio` | Advertising spend on radio (thousands $) |
| `newspaper` | Advertising spend on newspaper (thousands $) |
| `sales` | Product sales (thousands of units) — **target** |

**Download:** Run the first cell of `linear_regression.ipynb` or download from:  
https://www.statlearning.com/s/Advertising.csv

## Files

```
Linear_Regression/
├── README.md
├── linear_regression.ipynb
└── advertising.csv
```

## Usage

```python
from mlpackage.supervised_learning.linear_regression import LinearRegression

model = LinearRegression(method='ols')
model.fit(X_train, y_train)

print(f"R²:   {model.r_squared(X_test, y_test):.4f}")
print(f"RMSE: {model.rmse(X_test, y_test):.4f}")
```