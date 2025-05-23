# Unit-I Introduction: Neuron as basic unit of neurobiology, McCulloch-Pitts model, Hebbian Hypothesis; limitations of single-layered neural networks. 

import numpy as np

# --- McCulloch-Pitts Neuron Model ---
def mc_culloch_pitts(inputs, weights, threshold):
    summation = np.dot(inputs, weights)
    return 1 if summation >= threshold else 0

# Example: AND gate with McCulloch-Pitts neuron
print("McCulloch-Pitts AND gate:")
inputs_list = [(0, 0), (0, 1), (1, 0), (1, 1)]
weights = [1, 1]
threshold = 2
for inputs in inputs_list:
    output = mc_culloch_pitts(inputs, weights, threshold)
    print(f"Input: {inputs} => Output: {output}")

# --- Hebbian Learning Rule ---
# Hebb's Rule: Δw = η * x * y
def hebbian_learning(input_data, targets, eta=0.1):
    weights = np.zeros(input_data.shape[1])
    for x, y in zip(input_data, targets):
        weights += eta * x * y
    return weights

# Example: learning weights for OR logic
print("\nHebbian Learning for OR gate:")
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
targets = np.array([0,1,1,1])
learned_weights = hebbian_learning(inputs, targets)
print("Learned weights:", learned_weights)

# --- Limitation of Single-Layer Neural Networks: XOR Problem ---
def perceptron_train(X, y, eta=0.1, epochs=10):
    weights = np.zeros(X.shape[1])
    bias = 0
    for _ in range(epochs):
        for xi, target in zip(X, y):
            activation = np.dot(xi, weights) + bias
            output = 1 if activation > 0 else 0
            error = target - output
            weights += eta * error * xi
            bias += eta * error
    return weights, bias

def test_perceptron(X, weights, bias):
    outputs = []
    for xi in X:
        activation = np.dot(xi, weights) + bias
        output = 1 if activation > 0 else 0
        outputs.append(output)
    return outputs

# XOR dataset
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
y_xor = np.array([0,1,1,0])
weights, bias = perceptron_train(X_xor, y_xor)
outputs = test_perceptron(X_xor, weights, bias)

print("\nPerceptron on XOR (Single-layer limitation):")
for inp, out in zip(X_xor, outputs):
    print(f"Input: {inp} => Output: {out}")
