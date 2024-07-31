import numpy as np
import struct
import pickle

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
train_images_limited = train_images[:num_train_samples]
train_labels_limited = train_labels[:num_train_samples]

print("Limited Train Images Shape:", train_images_limited.shape)  
print("Limited Train Labels Shape:", train_labels_limited.shape)  

# k-NN algorithm
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_predict(train_images, train_labels, test_image, k=3):
    distances = np.array([euclidean_distance(test_image, x) for x in train_images])
    
    # Get the indices of the k nearest neighbors
    k_indices = distances.argsort()[:k]
    k_nearest_labels = train_labels[k_indices]
    
    unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
    return unique_labels[np.argmax(counts)]

def evaluate_knn(train_images, train_labels, test_images, test_labels, k=3):
    correct_predictions = 0
    num_test_samples = test_images.shape[0]

    for i in range(num_test_samples):
        test_image = test_images[i]
        true_label = test_labels[i]
        predicted_label = knn_predict(train_images, train_labels, test_image, k)

        if predicted_label == true_label:
            correct_predictions += 1

        if (i+1) % 1000 == 0:
            print(f"Processed {i+1} samples... Current Accuracy: {(correct_predictions / (i+1)) * 100:.2f}%")

    accuracy = correct_predictions / num_test_samples
    return accuracy

def save_knn_model(train_images, train_labels, k, filename='knn_model.pkl'):
    model_data = {
        'train_images': train_images,
        'train_labels': train_labels,
        'k': k
    }
    with open(filename, 'wb') as file:
        pickle.dump(model_data, file)
    
    print(f"Model saved to {filename}")

def load_knn_model(filename='knn_model.pkl'):
    with open(filename, 'rb') as file:
        model_data = pickle.load(file)
    
    print(f"Model loaded from {filename}")
    return model_data['train_images'], model_data['train_labels'], model_data['k']


k = 3  
save_knn_model(train_images_limited, train_labels_limited, k, 'knn_model_limited.pkl')
train_images_loaded, train_labels_loaded, k_loaded = load_knn_model('knn_model_limited.pkl')

print("Loaded Train Images Shape:", train_images_loaded.shape)
print("Loaded Train Labels Shape:", train_labels_loaded.shape)
print("Loaded k:", k_loaded)

accuracy_loaded = evaluate_knn(train_images_loaded, train_labels_loaded, test_images, test_labels, k_loaded)
print(f"Overall Accuracy: {accuracy_loaded * 100:.2f}%")