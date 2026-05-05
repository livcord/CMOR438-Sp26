import numpy as np

class MLP:
    """
    Multilayer Perceptron (MLP) for binary classification.

    A fully connected feedforward neural network trained using
    backpropagation and gradient descent.

    Architecture:
        - Hidden layers: ReLU activation
        - Output layer: Sigmoid activation (binary classification)

    Parameters
    layer_sizes : list of int
        List defining network architecture, e.g. [n_input, h1, h2, n_output].
    lr : float
        Learning rate for gradient descent.
    n_iters : int
        Number of training iterations.
    """

    def __init__(self, layer_sizes, lr=0.01, n_iters=1000):
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.n_iters = n_iters
        self.weights = []
        self.biases = []

        self._init_params()

    def _init_params(self):
        """
        Initialize weights and biases using small random values.

        Weights are initialized with a small Gaussian distribution
        to break symmetry, biases are initialized to zero.
        """
        np.random.seed(42)
        for i in range(len(self.layer_sizes) - 1):
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * 0.01
            b = np.zeros((1, self.layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    # Activations

    def _relu(self, Z):
        """
        ReLU activation function.

        Returns max(0, Z).
        """
        return np.maximum(0, Z)

    def _relu_derivative(self, Z):
        """
        Derivative of ReLU activation.

        Returns 1 for Z > 0 else 0.
        """
        return (Z > 0).astype(float)

    def _sigmoid(self, Z):
        """
        Sigmoid activation function.

        Maps values to range (0, 1).
        """
        return 1 / (1 + np.exp(-Z))

    def _sigmoid_derivative(self, A):
        """
        Derivative of sigmoid function.

        Assumes input is activation A = sigmoid(Z).
        """
        return A * (1 - A)


    def _forward(self, X):
        """
        Perform forward propagation through the network.

        Parameters
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        activations : list of np.ndarray
            Activations at each layer.
        Zs : list of np.ndarray
            Linear combinations at each layer.
        """
        activations = [X]
        Zs = []

        A = X
        for i in range(len(self.weights) - 1):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            A = self._relu(Z)
            Zs.append(Z)
            activations.append(A)

        # output layer
        Z = np.dot(A, self.weights[-1]) + self.biases[-1]
        A = self._sigmoid(Z)

        Zs.append(Z)
        activations.append(A)

        return activations, Zs


    def _compute_loss(self, y, y_pred):
        """
        Compute binary cross-entropy loss.

        Parameters
        y : np.ndarray
            True labels.
        y_pred : np.ndarray
            Predicted probabilities.

        Returns
        float
            Loss value.
        """
        eps = 1e-8
        return -np.mean(
            y * np.log(y_pred + eps) +
            (1 - y) * np.log(1 - y_pred + eps)
        )


    def _backward(self, activations, Zs, y):
        """
        Perform backpropagation to compute gradients.

        Parameters
        activations : list of np.ndarray
            Activations from forward pass.
        Zs : list of np.ndarray
            Linear outputs from forward pass.
        y : np.ndarray
            True labels.

        Returns
        grads_w : list of np.ndarray
            Gradients for weights.
        grads_b : list of np.ndarray
            Gradients for biases.
        """
        grads_w = []
        grads_b = []

        m = y.shape[0]
        y = y.reshape(-1, 1)

        # output layer gradient
        A_final = activations[-1]
        dZ = A_final - y

        for i in reversed(range(len(self.weights))):
            A_prev = activations[i]

            dW = (1 / m) * np.dot(A_prev.T, dZ)
            dB = (1 / m) * np.sum(dZ, axis=0, keepdims=True)

            grads_w.insert(0, dW)
            grads_b.insert(0, dB)

            if i != 0:
                dA_prev = np.dot(dZ, self.weights[i].T)
                dZ = dA_prev * self._relu_derivative(Zs[i - 1])

        return grads_w, grads_b

    # Train

    def fit(self, X, y):
        """
        Train the neural network using gradient descent.

        Parameters
        X : np.ndarray
            Training features.
        y : np.ndarray
            Training labels.

        Returns
        self
        """
        for _ in range(self.n_iters):
            activations, Zs = self._forward(X)
            grads_w, grads_b = self._backward(activations, Zs, y)

            grads_w = [np.clip(g, -5, 5) for g in grads_w]
            grads_b = [np.clip(g, -5, 5) for g in grads_b]

            for i in range(len(self.weights)):
                self.weights[i] -= self.lr * grads_w[i]
                self.biases[i] -= self.lr * grads_b[i]

        return self

    # Predict

    def predict(self, X):
        """
        Predict binary class labels.

        Parameters
        X : np.ndarray

        Returns
        np.ndarray
            Predicted class labels (0 or 1).
        """
        activations, _ = self._forward(X)
        probs = activations[-1]
        return (probs > 0.5).astype(int)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        X : np.ndarray

        Returns
        np.ndarray
            Probability of class 1.
        """
        activations, _ = self._forward(X)
        return activations[-1]
    
    def score(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)
        preds = self.predict(X)
        return np.mean(preds == y)