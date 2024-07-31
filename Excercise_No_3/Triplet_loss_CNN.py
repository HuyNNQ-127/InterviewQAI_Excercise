import numpy as np
import struct
import pickle
import time
start_time = time.time()

def read_mnist_images(filename):
    with open(filename, 'rb') as file:
        magic, num, rows, cols = struct.unpack(">IIII", file.read(16))
        images = np.fromfile(file, dtype=np.uint8).reshape(num, rows, cols)
    return images

def read_mnist_labels(file_path):
    with open(file_path, 'rb') as file:
        labels = np.fromfile(file, dtype=np.uint8)
    return labels

# Load data
train_images = read_mnist_images(r'E:\Python\QAI_Review_Test\mnist_dataset\train-images.idx3-ubyte')
train_labels = read_mnist_labels(r'E:\Python\QAI_Review_Test\mnist_dataset\train-labels.idx1-ubyte')
test_images = read_mnist_images(r'E:\Python\QAI_Review_Test\mnist_dataset\t10k-images.idx3-ubyte')
test_labels = read_mnist_labels(r'E:\Python\QAI_Review_Test\mnist_dataset\t10k-labels.idx1-ubyte')

# Preprocess data
X_train = train_images / 255.0
X_test = test_images / 255.0

X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

def triplet_loss(anchor, positive, negative, alpha=0.2):
    pos_dist = np.sum((anchor - positive) ** 2, axis=1)
    neg_dist = np.sum((anchor - negative) ** 2, axis=1)
    loss = np.maximum(0, pos_dist - neg_dist + alpha)
    return np.mean(loss)

class Triplet_loss_CNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def forward_pass(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2
    
    def compute_loss(self, anchor, positive, negative, alpha=0.2):
        anchor_output = self.forward(anchor)
        positive_output = self.forward(positive)
        negative_output = self.forward(negative)

        # Ensure there are no NaN values in outputs
        if np.isnan(anchor_output).any() or np.isnan(positive_output).any() or np.isnan(negative_output).any():
            print("NaN detected in forward pass outputs")
        
        loss = triplet_loss(anchor_output, positive_output, negative_output, alpha)
        return loss
    
    def backward(self, anchor, positive, negative, alpha=0.2, learning_rate=0.01):
        # forward pass
        anchor_output = self.forward_pass(anchor)
        positive_output = self.forward_pass(positive)
        negative_output = self.forward_pass(negative)

        pos_dist = 2 * (anchor_output - positive_output)
        neg_dist = 2 * (anchor_output - negative_output)
        
        dloss_da = pos_dist - neg_dist
        dloss_dp = -pos_dist
        dloss_dn = neg_dist

        if np.isnan(dloss_da).any() or np.isnan(dloss_dp).any() or np.isnan(dloss_dn).any():
            print("NaN in gradient!")

        # weights and bias
        self.W2 -= learning_rate * np.dot(self.a1.T, dloss_da)
        self.b2 -= learning_rate * np.sum(dloss_da, axis=0, keepdims=True)
        
        dW1_a = np.dot(anchor.T, np.dot(dloss_da, self.W2.T) * (self.z1 > 0))
        dW1_p = np.dot(positive.T, np.dot(dloss_dp, self.W2.T) * (self.z1 > 0))
        dW1_n = np.dot(negative.T, np.dot(dloss_dn, self.W2.T) * (self.z1 > 0))

        db1_a = np.sum(np.dot(dloss_da, self.W2.T) * (self.z1 > 0), axis=0, keepdims=True)            
        db1_p = np.sum(np.dot(dloss_dp, self.W2.T) * (self.z1 > 0), axis=0, keepdims=True)     
        db1_n = np.sum(np.dot(dloss_dn, self.W2.T) * (self.z1 > 0), axis=0, keepdims=True)
        
        self.W1 -= learning_rate * (dW1_a + dW1_p + dW1_n)
        self.b1 -= learning_rate * (db1_a + db1_p + db1_n)

def compute_accuracy(anchor, positive, negative):
    anchor_output = model.forward(anchor)
    positive_output = model.forward(positive)
    negative_output = model.forward(negative)
    
    pos_dist = np.sum(np.square(anchor_output - positive_output), axis=1)
    neg_dist = np.sum(np.square(anchor_output - negative_output), axis=1)
    
    correct = np.sum(pos_dist < neg_dist)
    accuracy = correct / len(pos_dist)
    
    return accuracy

#input_size, hidden_size, output_size
model = Triplet_loss_CNN(784, 128, 64)

permutation = np.random.permutation(X_train.shape[0])
X_train = X_train[permutation]
train_labels = train_labels[permutation]

batch_size = 64
num_epochs = 25

for epoch in range(num_epochs):
    learning_rate = 0.0001
    LOSS = []
    for i in range(0, X_train.shape[0], batch_size):
        end = i + batch_size
        if end > X_train.shape[0]:
            break
        
        anchor_batch = X_train[i:end]
        positive_batch = X_train[i:end]
        negative_batch = X_train[(i+batch_size) % X_train.shape[0]: (i+2*batch_size) % X_train.shape[0]]
        
        if len(negative_batch) < len(anchor_batch):
            continue
        
        loss = model.compute_loss(anchor_batch, positive_batch, negative_batch)
        LOSS.append(loss)
        model.backward(anchor_batch, positive_batch, negative_batch, learning_rate=learning_rate)
        
    print(f'Epoch {epoch+1}:')
    print(f'Loss: {np.mean(LOSS):.4f}')

with open('triplet_loss_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print('Model saved.')

anchor_test = X_test[:batch_size]
positive_test = X_test[:batch_size]
negative_test = X_test[batch_size:2*batch_size]

test_loss = model.compute_loss(anchor_test, positive_test, negative_test)
test_accuracy = compute_accuracy(anchor_test, positive_test, negative_test)

print(f'Accuracy: {test_accuracy * 100:.2f}%')
print("--- %s seconds ---" % (time.time() - start_time))