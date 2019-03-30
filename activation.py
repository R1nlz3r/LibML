from numba import jit
import numpy as np

def identity(x):
	return x

def identity_gradient(x):
	return 1

@jit
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

@jit
def sigmoid_gradient(x):
	return sigmoid(x) * (1 - sigmoid(x))

@jit
def tanh(x):
	return np.tanh(x)

@jit
def tanh_gradient(x):
	return 1.0 - tanh(x) ** 2

@jit
def softmax(x):
	return np.exp(x) / sum(np.exp(x))
