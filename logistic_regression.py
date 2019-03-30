import numpy as np
import activation as act

def adjust_weights(x, y, theta, alpha = 0.1, lmbd = 1, activation = 'sigmoid'):
	regularized_theta = theta.copy()
	regularized[0] = 0
	m = x.shape[0]

	act_function, act_gradient = act.distribute_functions(activation)

	theta_gradient = (np.dot(x.T, (sigmoid(np.dot(x, theta)) - y)) + lmbd * regularized_theta)
	theta -= (theta_gradient * alpha) / m

	return theta
