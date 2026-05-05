# Perceptron

## Overview

The **Perceptron** is the simplest possible neural network and the historical origin of all modern deep learning. Proposed in 1958, it is a single-layer linear binary classifier that learns exclusively by correcting its own mistakes, one sample at a time.

## How It Works

The Perceptron computes a weighted sum of inputs and fires based on a step threshold:

$$\hat{y} = \begin{cases} 1 & \text{if } \mathbf{w} \cdot \mathbf{x} + b \geq 0 \\ 0 & \text{otherwise} \end{cases}$$

For each sample, if the prediction is correct nothing happens. If it is wrong, the weight vector is nudged toward the correct answer:

$$\mathbf{w} \leftarrow \mathbf{w} + \eta\,(y - \hat{y})\,\mathbf{x}, \qquad b \leftarrow b + \eta\,(y - \hat{y})$$

where $\eta$ is the learning rate. This is called the **Perceptron learning rule**.

## The Convergence Theorem

The **Perceptron Convergence Theorem** states that if the training data is linearly separable, the algorithm is guaranteed to find a separating hyperplane in a finite number of steps. If the data is *not* linearly separable, the algorithm cycles indefinitely. This is a fundamental limitation that motivates Multilayer Perceptrons (MLPs).

## Key Properties

| Property | Detail |
|---|---|
| Type | Supervised — Binary Classification |
| Decision boundary | Linear hyperplane |
| Convergence | Guaranteed only when data is linearly separable |
| Feature scaling | **Required**:mthe dot product is dominated by large-magnitude features |
| Output | Hard label (0 or 1), no probability estimate |

## Dataset

**Pima Indians Diabetes** — National Institute of Diabetes and Digestive and Kidney Diseases.

| Feature | Description |
|---|---|
| `pregnancies` | Number of pregnancies |
| `glucose` | Plasma glucose concentration |
| `blood_pressure` | Diastolic blood pressure (mm Hg) |
| `skin_thickness` | Triceps skin fold thickness (mm) |
| `insulin` | 2-hour serum insulin (mu U/ml) |
| `bmi` | Body mass index |
| `diabetes_pedigree` | Diabetes pedigree function |
| `age` | Age in years |
| `outcome` | Target: 0 = no diabetes, 1 = diabetes |

**Download:** First notebook cell downloads automatically from UCI. The dataset is also attached to this subfolder for direct download.

## Files

```
Perceptron/
├── README.md
├── perceptron.ipynb
└── diabetes.csv
```

## Usage

```python
from mlpackage.supervised_learning.perceptron import Perceptron
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = Perceptron(learning_rate=0.01, n_iterations=1000)
model.fit(X_train_s, y_train)
print(f"Accuracy: {model.score(X_test_s, y_test):.4f}")
```