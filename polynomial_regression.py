import numpy as np

def prediction(theta, x):
	total = 0
	for degree in range(0, theta.shape[1]):
		total += theta[0, degree] * x ** degree
	return total


# Features need to be standardized
def adjust_weights(x, y, theta, alpha = 0.1):
	m = len(x)
	tmp_theta = np.zeros((theta.shape))
	for i in range(0, m):
		for degree in range(0, theta.shape[1]):
			tmp_theta[0, degree] += ((prediction(theta, x[i]) - y[i]) * x[i] ** degree)
	theta -= (tmp_theta * alpha) / m

	return theta
