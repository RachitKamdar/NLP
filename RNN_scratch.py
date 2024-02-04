import numpy as np

def step(s, x, U, W):
	return x * U + s * W


def forward(x, U, W):
	# Number of samples in the mini-batch
	number_of_samples = len(x)
	# Length of each sample
	sequence_length = len(x[0])
	# Initialize the state activation for each sample along the sequence
	s = np.zeros((number_of_samples, sequence_length + 1))
	# Update the states over the sequence
	for t in range(0, sequence_length):
		s[:, t + 1] = step(s[:, t], x[:, t], U, W) # step function
	return s

def backward(x, s, y, W):
	sequence_length = len(x[0])
	# The network output is just the last activation of sequence
	s_t = s[:, -1]
	# Compute the gradient of the output w.r.t. MSE loss function at final state
	gS = 2 * (s_t - y)
	# Set the gradient accumulations to 0
	gU, gW = 0, 0
	# Accumulate gradients backwards
	for k in range(sequence_length, 0, -1):
	# Compute the parameter gradients and accumulate theresults
		gU += np.sum(gS * x[:, k - 1])
		gW += np.sum(gS * s[:, k - 1])
		# Compute the gradient at the output of the previous layer
		gS = gS * W
	return gU, gW

def train(x, y, epochs, learning_rate=0.0005):
	weights = (-2, 0) # (U, W)
	# Accumulate the losses and their respective weights
	losses = list()
	gradients_u = list()
	gradients_w = list()
	# Perform iterative gradient descent
	for i in range(epochs):
		# Perform forward and backward pass to get the gradients
		s = forward(x, weights[0], weights[1])
		# Compute the loss
		loss = (y[0] - s[-1, -1]) ** 2
		# Store the loss and weights values for later display
		losses.append(loss)
		gradients = backward(x, s, y, weights[1])
		gradients_u.append(gradients[0])
		gradients_w.append(gradients[1])
		weights = tuple((p - gp * learning_rate) for p, gp in zip(weights, gradients))
	print(weights)
	return np.array(losses), np.array(gradients_u),np.array(gradients_w)



if __name__ == '__main__':
	x = np.array([[0, 0, 0, 0, 1, 0, 1, 0, 1, 0]])
	y = np.array([3])
	losses, gradients_u, gradients_w = train(x, y, epochs=150)
	print(losses)