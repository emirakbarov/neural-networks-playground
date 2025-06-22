import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load MNIST
def load_mnist():
    mnist = fetch_openml('mnist_784', version=1)
    data = mnist['data'].to_numpy() / 255.0 
    labels = mnist['target'].astype(int)

    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size), labels] = 1

    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(
        data, one_hot_labels, test_size=0.2, random_state=42
    )

    return x_train.T, y_train.T, x_test.T, y_test.T

# Activation Funcs
def sigmoid(z):
    return 1 / (1 + np.exp(-z)) # Sigmoid Activation Function

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z)) # Derivative of the Sigmoid for Backprop

# Initialize Params 
def init_network(sizes):
    weights = [np.random.randn(sizes[i+1], sizes[i]) * np.sqrt(1 / sizes[i]) for i in range(len(sizes) - 1)] # Initialize random weights
    biases = [np.zeros((sizes[i+1], 1)) for i in range(len(sizes) - 1)] # Initialise random biases
    return weights, biases 

#  Forward Propagation 
def feedforward(x, weights, biases):
    activations = [x] # Layer of activations 
    zs = [] # Layer of preactivations

    for w, b in zip(weights, biases):
        z = np.dot(w, activations[-1]) + b # Calculate the preactivation using the dot product
        zs.append(z) # add it to the layer of preactivations
        activations.append(sigmoid(z)) # Activate and add to the layer

    return activations, zs

# Backpropagation 
def backprop(x, y, weights, biases): # Nabla = ∇
    nabla_w = [np.zeros_like(w) for w in weights] # Differentiate cost with respect to weight
    nabla_b = [np.zeros_like(b) for b in biases] # Differentiate cost with respect to bias

    # Forward pass
    activations, zs = feedforward(x, weights, biases)

    # Output layer error
    delta = (activations[-1] - y) * sigmoid_prime(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].T)

    # Backpropagate the error
    for l in range(2, len(weights) + 1):
        z = zs[-l] 
        sp = sigmoid_prime(z)
        delta = np.dot(weights[-l + 1].T, delta) * sp
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot(delta, activations[-l - 1].T)

    return nabla_w, nabla_b

# --- Update mini-batch ---
def update_mini_batch(x_batch, y_batch, weights, biases, eta):
    nabla_w = [np.zeros_like(w) for w in weights]
    nabla_b = [np.zeros_like(b) for b in biases]

    for x, y in zip(x_batch.T, y_batch.T):
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        delta_w, delta_b = backprop(x, y, weights, biases)
        nabla_w = [nw + dw for nw, dw in zip(nabla_w, delta_w)]
        nabla_b = [nb + db for nb, db in zip(nabla_b, delta_b)]

    weights = [w - (eta / x_batch.shape[1]) * nw for w, nw in zip(weights, nabla_w)]
    biases = [b - (eta / x_batch.shape[1]) * nb for b, nb in zip(biases, nabla_b)]

    return weights, biases

# --- Evaluate Accuracy ---
def evaluate(x_test, y_test, weights, biases):
    correct = 0
    for i in range(x_test.shape[1]):
        x = x_test[:, i].reshape(-1, 1)
        y = y_test[:, i].reshape(-1, 1)
        output = feedforward(x, weights, biases)[0][-1]
        prediction = np.argmax(output)
        actual = np.argmax(y)
        correct += (prediction == actual)
    return correct

# --- Train Network ---
def train_network(x_train, y_train, x_test, y_test, sizes, epochs=10, batch_size=32, eta=3.0):
    weights, biases = init_network(sizes)
    n = x_train.shape[1]

    for epoch in range(epochs):
        permutation = np.random.permutation(n)
        x_shuffled = x_train[:, permutation]
        y_shuffled = y_train[:, permutation]

        for k in range(0, n, batch_size):
            x_batch = x_shuffled[:, k:k + batch_size]
            y_batch = y_shuffled[:, k:k + batch_size]
            weights, biases = update_mini_batch(x_batch, y_batch, weights, biases, eta)

        acc = evaluate(x_test, y_test, weights, biases)
        print(f"Epoch {epoch+1}: {acc} / {x_test.shape[1]} accuracy = {acc / x_test.shape[1] * 100:.2f}%")

    return weights, biases

# --- Run Everything ---
if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_mnist()
    sizes = [784, 64, 10]  
    train_network(x_train, y_train, x_test, y_test, sizes)
