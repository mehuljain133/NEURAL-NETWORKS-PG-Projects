# Unit-II Supervised Learning: Single-layered neural networks, perceptron rule, review of gradientdescent algorithms; multi-layered neural networks: first order methods, backpropagation algorithm, second order methods, modelling sequences using recurrent neural networks, Hopefield networks, Boltzmann machines, restricted Boltzmann machines

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. Single-layer Perceptron (Perceptron Rule) ---
class Perceptron:
    def __init__(self, input_size, lr=0.1):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.lr = lr

    def predict(self, x):
        activation = np.dot(x, self.weights) + self.bias
        return 1 if activation >= 0 else 0

    def train(self, X, y, epochs=10):
        for _ in range(epochs):
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction
                self.weights += self.lr * error * xi
                self.bias += self.lr * error

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,1])  # OR gate
p = Perceptron(input_size=2)
p.train(X, y)
print("\nPerceptron (OR Gate):")
for xi in X:
    print(f"{xi} => {p.predict(xi)}")

# --- 2. Gradient Descent (Manual) ---
def gradient_descent_demo():
    w = 0.0
    lr = 0.01
    for i in range(100):
        loss = (w - 3)**2
        grad = 2 * (w - 3)
        w -= lr * grad
    print("\nManual Gradient Descent: Final w =", w)

gradient_descent_demo()

# --- 3. Multi-layer Neural Network (Backpropagation with PyTorch) ---
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(x))))

model = MLP()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y.reshape(-1,1), dtype=torch.float32)

for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

print("\nMulti-layer Network (OR Gate):")
with torch.no_grad():
    print(torch.round(model(X_tensor).view(-1)))

# --- 4. Recurrent Neural Network (RNN for Sequence Modeling) ---
class SimpleRNN(nn.Module):
    def __init__(self):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=5, batch_first=True)
        self.fc = nn.Linear(5, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

rnn = SimpleRNN()
optimizer = optim.Adam(rnn.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Sequence: input [1,2,3] => target [4]
seq_in = torch.tensor([[[1.],[2.],[3.]]])
seq_out = torch.tensor([[4.]])

for i in range(300):
    pred = rnn(seq_in)
    loss = criterion(pred, seq_out)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("\nRNN Sequence Prediction: [1,2,3] =>", rnn(seq_in).item())

# --- 5. Hopfield Network ---
class Hopfield:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        for p in patterns:
            self.weights += np.outer(p, p)
        np.fill_diagonal(self.weights, 0)

    def recall(self, pattern, steps=5):
        s = pattern.copy()
        for _ in range(steps):
            for i in range(self.size):
                s[i] = 1 if np.dot(self.weights[i], s) >= 0 else -1
        return s

print("\nHopfield Network:")
patterns = [np.array([1,-1,1,-1])]
hop = Hopfield(4)
hop.train(patterns)
print("Recall:", hop.recall(np.array([1,-1,-1,-1])))

# --- 6. Restricted Boltzmann Machine (RBM using PyTorch) ---
class RBM(nn.Module):
    def __init__(self, n_vis, n_hid):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hid, n_vis) * 0.1)
        self.h_bias = nn.Parameter(torch.zeros(n_hid))
        self.v_bias = nn.Parameter(torch.zeros(n_vis))

    def sample_h(self, v):
        prob = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return torch.bernoulli(prob)

    def sample_v(self, h):
        prob = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return torch.bernoulli(prob)

import torch.nn.functional as F
rbm = RBM(n_vis=6, n_hid=2)
data = torch.bernoulli(torch.rand(10, 6))

# One step of Contrastive Divergence
v0 = data
h0 = rbm.sample_h(v0)
v1 = rbm.sample_v(h0)
h1 = rbm.sample_h(v1)

rbm.W.data += 0.01 * (torch.mm(h0.t(), v0) - torch.mm(h1.t(), v1))
print("\nRBM weight update done.")

