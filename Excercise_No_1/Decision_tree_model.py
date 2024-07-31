import numpy as np
import struct

def read_mnist_images(filename):
    with open(filename, 'rb') as file:
        magic, num, rows, cols = struct.unpack(">IIII", file.read(16))
        images = np.fromfile(file, dtype=np.uint8).reshape(num, rows, cols)
    return images

def read_mnist_labels(file_path):
    with open(file_path, 'rb') as file:
        labels = np.fromfile(file, dtype=np.uint8)
    return labels

train_images = read_mnist_images('mnist_dataset/train-images.idx3-ubyte')
train_labels = read_mnist_labels('mnist_dataset/train-labels.idx1-ubyte')
test_images = read_mnist_images('mnist_dataset/t10k-images.idx3-ubyte')
test_labels = read_mnist_labels('mnist_dataset/t10k-labels.idx1-ubyte')

train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

train_images = train_images.reshape(train_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)

num_train_samples = 10000
train_images = train_images[:num_train_samples]
train_labels = train_labels[:num_train_samples]

def pca(X, num_components):
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    covariance_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    principal_components = eigenvectors[:, :num_components]
    X_pca = np.dot(X_centered, principal_components)
    return X_pca, principal_components, mean

n_components = 25
train_images_pca, pca_components, pca_mean = pca(train_images, n_components)
test_images_pca = np.dot(test_images - pca_mean, pca_components)

class DecisionTree:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.tree = None
    
    def fit(self, X, y):
        self.tree = self._fit(X, y, depth=0)
    
    def _fit(self, X, y, depth):
        num_samples, num_features = X.shape
        unique_labels = np.unique(y)
        
        if len(unique_labels) == 1:
            return unique_labels[0]
        
        if depth == self.max_depth:
            return np.bincount(y).argmax()
        
        best_gini = float('inf')
        best_split = None
        best_left_indices = None
        best_right_indices = None
        
        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = np.where(X[:, feature] <= threshold)[0]
                right_indices = np.where(X[:, feature] > threshold)[0]
                
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                
                gini_left = self._gini_index(y[left_indices])
                gini_right = self._gini_index(y[right_indices])
                gini_split = (len(left_indices) * gini_left + len(right_indices) * gini_right) / num_samples
                
                if gini_split < best_gini:
                    best_gini = gini_split
                    best_split = (feature, threshold)
                    best_left_indices = left_indices
                    best_right_indices = right_indices
        
        if best_split is None:
            return np.bincount(y).argmax()
        
        left_tree = self._fit(X[best_left_indices], y[best_left_indices], depth + 1)
        right_tree = self._fit(X[best_right_indices], y[best_right_indices], depth + 1)
        
        return (best_split, left_tree, right_tree)
    
    def _gini_index(self, y):
        unique_labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini
    
    def predict(self, X):
        return np.array([self._predict(x, self.tree) for x in X])
    
    def _predict(self, x, tree):
        if not isinstance(tree, tuple):
            return tree
        
        feature, threshold = tree[0]
        if x[feature] <= threshold:
            return self._predict(x, tree[1])
        else:
            return self._predict(x, tree[2])

clf = DecisionTree(max_depth=10)
clf.fit(train_images_pca, train_labels)

test_predictions = clf.predict(test_images_pca)

accuracy = np.mean(test_predictions == test_labels)
print(f"Accuracy with PCA: {accuracy * 100:.2f}%")
