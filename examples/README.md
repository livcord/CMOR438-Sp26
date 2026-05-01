# Machine Learning From Scratch — Examples

This directory contains implementations of core machine learning algorithms built from scratch. The focus is on understanding how each model works internally and how well it behaves on real or synthetic data.

Each subfolder includes:
- A **Jupyter notebook** demonstrating the algorithm step by step  
- A **dataset** (real or generated) used in the example  
- A short **README** explaining the intuition, implementation, and results  

The examples are organized into **Supervised Learning** and **Unsupervised Learning**.

---

## Supervised Learning

Supervised learning uses labeled data, where each input is paired with a known output. The goal is to learn a mapping from inputs to outputs that generalizes well to new data.

### Common Tasks
- **Classification** — Predict discrete labels  
- **Regression** — Predict continuous values  

### Implemented Algorithms

**Decision Trees**  
Builds a tree of decision rules by recursively splitting the data.

**Ensemble Methods**  
Combine multiple models (e.g., averaging or voting) to improve performance, stability, and generalization compared to a single model.

**K-Nearest Neighbors (KNN)**  
Predicts outcomes based on the nearest data points in feature space.

**Linear Regression**  
Fits a linear relationship between features and a continuous target.

**Logistic Regression**  
Models probabilities for binary classification using a sigmoid function.

**Multilayer Perceptron (MLP)**  
A feedforward neural network with one or more hidden layers that can learn non-linear relationships using activation functions and backpropagation.

**Perceptron**  
A simple linear classifier that updates weights based on prediction errors.

**Random Forests**  
An ensemble of decision trees trained on different subsets of the data and features; predictions are aggregated (e.g., by voting or averaging) to reduce overfitting and improve robustness.

---

### Key Characteristics
- Requires labeled training data  
- Focused on predictive performance  
- Evaluated using metrics such as accuracy, RMSE, and R²  

---

## Unsupervised Learning

Unsupervised learning works with unlabeled data. Instead of predicting outputs, it aims to uncover structure, groupings, or lower-dimensional representations.

### Common Tasks
- **Clustering** — Group similar data points  
- **Dimensionality Reduction** — Reduce feature space while preserving structure  

### Implemented Algorithms

**DBSCAN**  
Finds clusters based on density and identifies noise points.

**K-Means**  
Partitions data into clusters by minimizing within-cluster variance.

**PCA (Principal Component Analysis)**  
Projects data onto directions that capture the most variance.

---

### Key Characteristics
- Does not require labels  
- Often used for exploration or preprocessing  
- Helps reveal hidden structure in data  