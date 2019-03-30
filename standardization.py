from numpy import mean, std
from numba import jit

@jit
def standardize(x):
	return (x - mean(x)) / std(x)

@jit
def destandardize(x, x_ref):
	return x * std(x_ref) + mean(x_ref)
