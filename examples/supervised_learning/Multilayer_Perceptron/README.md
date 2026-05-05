# Multilayer Perceptron (MLP)

## Overview

The **Multilayer Perceptron** is a fully connected feedforward neural network that extends the single-layer Perceptron by stacking multiple layers of neurons with non-linear activation functions. This depth allows the MLP to learn arbitrarily complex decision boundaries, overcoming the Perceptron's fundamental limitation of linear separability.

## How It Works

### Architecture

An MLP consists of:
- **Input layer**: one neuron per feature
- **Hidden layers**: one or more layers, each applying a linear transformation followed by a non-linear activation
- **Output layer**: one neuron per class (softmax for multiclass, sigmoid for binary)

For each layer $l$:

$$\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}, \qquad \mathbf{a}^{(l)} = \phi\!\left(\mathbf{z}^{(l)}\right)$$

### Activation Functions

| Function | Formula | Use |
|---|---|---|
| ReLU | $\max(0, z)$ | Hidden layers — efficient, no vanishing gradient |
| Sigmoid | $1/(1+e^{-z})$ | Binary output — squashes to (0,1) |
| Softmax | $e^{z_k}/\sum_j e^{z_j}$ | Multiclass output — normalizes to probability distribution |

### Backpropagation

Training uses **backpropagation** with **gradient descent**: the chain rule propagates the gradient of the loss backward through every layer, computing how each weight contributed to the error. Weights are updated to reduce the loss:

$$\mathbf{W}^{(l)} \leftarrow \mathbf{W}^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}}$$

### Loss Functions

- **Cross-entropy** (classification): $\mathcal{L} = -\sum_k y_k \log \hat{p}_k$
- **MSE** (regression): $\mathcal{L} = \frac{1}{n}\sum(y - \hat{y})^2$

## Key Properties

| Property | Detail |
|---|---|
| Type | Supervised — Classification or Regression |
| Decision boundary | Non-linear (can approximate any continuous function) |
| Feature scaling | **Required** — gradient descent is sensitive to feature magnitude |
| Key hyperparameters | Layer sizes, learning rate, activation, epochs |
| Universal approximation | An MLP with one hidden layer can approximate any continuous function (given sufficient neurons) |

## Dataset

**Handwritten Digits (sklearn Digits)** — 8×8 pixel grayscale images of digits 0–9.

| Detail | Value |
|---|---|
| Samples | 1,797 |
| Features | 64 (8×8 pixel values, range 0–16) |
| Classes | 10 (digits 0–9) |
| Source | sklearn.datasets.load_digits |

**Download:** First notebook cell exports the dataset from sklearn to CSV automatically. The dataset is also attached to this subfolder.

## Files

```
Multilayer_Perceptron/
├── README.md
├── mlp.ipynb
└── digits.csv
```

## Usage

```python
from mlpackage.supervised_learning.mlp import MLP
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = MLP(hidden_layers=[128, 64], learning_rate=0.01,
            n_iterations=500, activation='relu')
model.fit(X_train_s, y_train)
print(f"Accuracy: {model.score(X_test_s, y_test):.4f}")
```