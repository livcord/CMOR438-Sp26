# Random Forest

## Overview

A **Random Forest** is an ensemble of Decision Trees that corrects the single biggest flaw of individual trees: **high variance**. A single deep tree is extremely sensitive to the training data. If we change a few samples, you get a completely different tree. Random Forests average the predictions of many trees, each trained with deliberate randomness, to produce a stable, accurate model.

## How It Works

Two sources of randomness are injected to decorrelate the trees:

**1. Bootstrap Sampling (Bagging)**  
Each tree is trained on a different random sample of the training data, drawn *with replacement*. On average, each bootstrap sample contains ~63% of the original data. The remaining ~37% are called **out-of-bag (OOB)** samples.

**2. Feature Subsampling**  
At each split, only a random subset of features is considered (typically $\sqrt{d}$ for classification). This forces trees to be diverse, they cannot all learn the same dominant features.

**Aggregation**  
- Classification: majority vote across all trees → $\hat{y} = \text{mode}\bigl(\{\hat{y}_1, \ldots, \hat{y}_T\}\bigr)$  
- Regression: mean across all trees → $\hat{y} = \frac{1}{T}\sum_{t=1}^T \hat{y}_t$

## The Law of Large Numbers Behind Forests

If individual trees have error rate $\epsilon$ and are independent, the ensemble error decreases as $T \to \infty$. Perfect independence is not achievable (trees share training data and features), but the two randomness sources reduce correlation substantially and the variance reduction effect is real.

## Out-of-Bag Score

Because each tree only sees ~63% of the data, the remaining 37% (OOB samples) act as a free validation set. The OOB score is an unbiased estimate of generalization error, computed at no extra cost.

## Key Properties

| Property | Detail |
|---|---|
| Type | Supervised — Classification or Regression |
| Base learner | Decision Tree |
| Parallelizable | Yes — trees are independent |
| Feature scaling | Not required (trees use threshold splits) |
| Key strength | Low variance, robust to noise and outliers |
| Key weakness | Less interpretable than a single tree |

## Dataset

**Heart Disease (Cleveland)** — UCI Machine Learning Repository.

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | Sex (1=male, 0=female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 (1=true) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels (0–3) |
| `thal` | Thalassemia (1=normal, 2=fixed, 3=reversible defect) |
| `target` | 0 = no disease, 1 = disease |

**Download:** First notebook cell downloads automatically from UCI. The dataset is also attached to this subfolder.

## Files

```
Random_Forests/
├── README.md
├── random_forest.ipynb
└── heart_disease.csv
```

## Usage

```python
from mlpackage.supervised_learning.random_forest import RandomForest

model = RandomForest(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

print(f"Test accuracy : {model.score(X_test, y_test):.4f}")
print(f"OOB score     : {model.oob_score_:.4f}")
```