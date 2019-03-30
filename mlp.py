import numpy as np
import activation as act

def cost(a, y, m, nb_layers):
	return (-(np.dot(y.T, np.log(a[nb_layers])) + \
		np.dot(1 - y.T, np.log(1 - a[nb_layers]))) / m)[0][0]


def feed_forward(x, theta, act_function):
	m = x.shape[0]
	nb_layers = len(theta)

	# Add bias and initialize neurons
	a = [np.hstack((np.ones((m, 1)), x))]
	z = [np.dot(a[0], theta[0].T)]

	# Propagate sigmoid computation on each layer one by one
	for cur_layer in range(0, nb_layers - 1):
		a.extend([np.hstack((np.ones((len(z[cur_layer]), 1)), act_function(z[cur_layer])))])
		z.extend([np.dot(a[cur_layer + 1], theta[cur_layer + 1].T)])

	# Compute output layer
	a.extend([np.apply_along_axis(act.softmax, axis = 1, arr = z[nb_layers - 1])])

	return a, z

def back_propagation(a, z, y, theta, act_gradient):
	nb_layers = len(theta)
	d = [a[nb_layers] - y]

	# Backpropagate gradients
	for cur_layer in range(0, nb_layers - 1):
		d.extend([np.dot(d[cur_layer], theta[-(cur_layer + 1)][:, 1:]) * \
			act_gradient(z[-(cur_layer + 2)])])

	return d


def gradient_update(a, d, m, theta, alpha = 0.1, lmbd = 1):
	theta_gradient = []
	nb_layers = len(theta)

	for cur_layer in range(0, nb_layers):
		theta_gradient.extend([np.dot(d[-(cur_layer + 1)].T, a[cur_layer]) / m])
		theta_gradient[cur_layer][:, 1:] += (lmbd * theta[cur_layer][:, 1:]) / m
		# Adjust to the learning rate
		theta[cur_layer] -= alpha * theta_gradient[cur_layer]

	return theta


def adjust_weights(x, y, theta, alpha = 0.1, lmbd = 1, activation = 'sigmoid'):
	m = x.shape[0]
	nb_layers = len(theta)

	# Choose activation function to compute hidden layers
	act_function, act_gradient = act.distribute_functions(activation)

	a, z = feed_forward(x, theta, act_function)
	d = back_propagation(a, z, y, theta, act_gradient)
	theta = gradient_update(a, d, m, theta, alpha, lmbd)

	return theta, a
