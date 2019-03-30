from numpy import argmax, log

def cross_entropy(y, output_layer):
	total = 0

	for i in range(0, output_layer.shape[0]):
		for j in range(0, output_layer.shape[1]):
			if y[i][j] == 1:
				total += log(output_layer[i][j])
			else:
				total += log(1 - output_layer[i][j])

	return - (total / (output_layer.shape[0] * output_layer.shape[1]))


def f1_score(y, output_layer):
	true = 0
	false = 0

	for i in range(0, len(output_layer)):
		if y[i, argmax(output_layer, axis = 1)[i]] == 1:
			true += 1.
		else:
			false += 1.

	return false / (true + false)
