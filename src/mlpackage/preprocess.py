import numpy as np

# Train/Test Split

def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
    """
    Split dataset into train and test sets
    """
    if random_state is not None:
        np.random.seed(random_state)

    X = np.array(X)
    y = np.array(y)

    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    test_count = int(n_samples * test_size)

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    return (
        X[train_indices],
        X[test_indices],
        y[train_indices],
        y[test_indices],
    )

# Standard Scaler

class StandardScaler:
    """
    Standardize features: (X - mean) / std
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        X = np.array(X)

        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        return self

    def transform(self, X):
        X = np.array(X)
        return (X - self.mean) / (self.std + 1e-8)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

# Min-Max Scaler

class MinMaxScaler:
    """
    Scale features to [0, 1]
    """

    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, X):
        X = np.array(X)

        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)

        return self

    def transform(self, X):
        X = np.array(X)
        return (X - self.min) / (self.max - self.min + 1e-8)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

# One-Hot Encoding

def one_hot_encode(y):
    """
    Convert categorical labels to one-hot encoding
    """
    y = np.array(y)
    classes = np.unique(y)

    class_to_index = {cls: i for i, cls in enumerate(classes)}

    one_hot = np.zeros((len(y), len(classes)))

    for i, label in enumerate(y):
        one_hot[i, class_to_index[label]] = 1

    return one_hot