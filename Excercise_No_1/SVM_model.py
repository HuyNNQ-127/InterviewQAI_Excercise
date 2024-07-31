import struct
import numpy as np

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

train_images = train_images.reshape(train_images.shape[0], -1) / 255.0
test_images = test_images.reshape(test_images.shape[0], -1) / 255.0

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        # Training
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.learning_rate * y[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

class OneVsAllSVM:
    def __init__(self, n_classes, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.n_classes = n_classes
        self.models = [SVM(learning_rate, lambda_param, n_iters) for _ in range(n_classes)]

    def fit(self, X, y):
        for i in range(self.n_classes):
            print(f"Training digit: {i}...")
            y_binary = np.where(y == i, 1, -1)
            self.models[i].fit(X, y_binary)

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])

        return np.argmax(predictions, axis=0)

ova_svm = OneVsAllSVM(n_classes=10, learning_rate=0.001, lambda_param=0.01, n_iters=1000)
ova_svm.fit(train_images, train_labels)

predictions = ova_svm.predict(test_images)

accuracy = np.mean(predictions == test_labels)
print(f"SVM One-vs-All Accuracy: {accuracy * 100:.2f}%")