# iris_numpy_nn.py
# NumPy-only 2-layer neural network for Iris classification (3 classes)
# Usage: python iris_numpy_nn.py

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

np.random.seed(42)

# Load and preprocess
iris = datasets.load_iris()
X = iris.data.astype(np.float32)  # (150,4)
y = iris.target.reshape(-1,1)     # (150,1)

# One-hot encode labels
enc = OneHotEncoder(sparse=False)
Y = enc.fit_transform(y)  # (150,3)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Network architecture
D = X_train.shape[1]
H = 16
C = Y_train.shape[1]

# Initialize weights
def glorot_init(in_dim, out_dim):
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    return np.random.uniform(-limit, limit, (in_dim, out_dim)).astype(np.float32)

W1 = glorot_init(D, H)
b1 = np.zeros((1, H), dtype=np.float32)
W2 = glorot_init(H, C)
b2 = np.zeros((1, C), dtype=np.float32)

# Activations and helper functions
def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(np.float32)

def cross_entropy_loss(probs, targets):
    # targets: one-hot
    N = targets.shape[0]
    loss = -np.sum(targets * np.log(probs + 1e-12)) / N
    return loss

# Training loop (mini-batch SGD)
lr = 0.01
epochs = 1000
batch_size = 16
N = X_train.shape[0]

for epoch in range(1, epochs+1):
    # Shuffle
    perm = np.random.permutation(N)
    X_sh, Y_sh = X_train[perm], Y_train[perm]
    for i in range(0, N, batch_size):
        xb = X_sh[i:i+batch_size]
        yb = Y_sh[i:i+batch_size]

        # forward
        z1 = xb.dot(W1) + b1         # (b,H)
        a1 = relu(z1)                # (b,H)
        z2 = a1.dot(W2) + b2         # (b,C)
        probs = softmax(z2)          # (b,C)

        # loss
        loss = cross_entropy_loss(probs, yb)

        # backward
        Nbatch = xb.shape[0]
        dz2 = (probs - yb) / Nbatch  # (b,C)
        dW2 = a1.T.dot(dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2.dot(W2.T)          # (b,H)
        dz1 = da1 * relu_grad(z1)
        dW1 = xb.T.dot(dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # update
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

    if epoch % 100 == 0 or epoch==1:
        # evaluate train accuracy
        a1_train = relu(X_train.dot(W1) + b1)
        probs_train = softmax(a1_train.dot(W2) + b2)
        preds_train = np.argmax(probs_train, axis=1)
        acc_train = np.mean(preds_train == np.argmax(Y_train, axis=1))
        print(f"Epoch {epoch:4d}  Loss {loss:.4f}  Train acc {acc_train:.4f}")

# Final test accuracy
a1_test = relu(X_test.dot(W1) + b1)
probs_test = softmax(a1_test.dot(W2) + b2)
preds_test = np.argmax(probs_test, axis=1)
acc_test = np.mean(preds_test == np.argmax(Y_test, axis=1))
print("Test accuracy:", acc_test)
