import numpy as np
from . import config as C

W1_SHAPE = (C.N_H1, C.N_X)
W2_SHAPE = (C.N_H2, C.N_H1)
W3_SHAPE = (C.N_Y, C.N_H2)

NUM_WEIGHTS = W1_SHAPE[0]*W1_SHAPE[1] + W2_SHAPE[0]*W2_SHAPE[1] + W3_SHAPE[0]*W3_SHAPE[1]

def split_weights(flat):
    """Decode flat genome -> (W1, W2, W3)."""
    a = W1_SHAPE[0]*W1_SHAPE[1]
    b = a + W2_SHAPE[0]*W2_SHAPE[1]
    W1 = flat[:a].reshape(W1_SHAPE)
    W2 = flat[a:b].reshape(W2_SHAPE)
    W3 = flat[b:].reshape(W3_SHAPE)
    return W1, W2, W3

def softmax(z):
    z = z - np.max(z, axis=0, keepdims=True)  # stability
    e = np.exp(z)
    return e / np.sum(e, axis=0, keepdims=True)

def forward(X, flat):
    """X: (batch, 7) -> probs: (batch, 3) using column-major math"""
    W1, W2, W3 = split_weights(flat)
    Xc = X.T  # (7, batch)
    Z1 = W1 @ Xc
    A1 = np.tanh(Z1)
    Z2 = W2 @ A1
    A2 = np.tanh(Z2)
    Z3 = W3 @ A2
    A3 = softmax(Z3)        # (3, batch)
    return A3.T             # (batch, 3)

def random_genome(rng):
    return rng.uniform(-1.0, 1.0, size=(NUM_WEIGHTS,)).astype(np.float32)
