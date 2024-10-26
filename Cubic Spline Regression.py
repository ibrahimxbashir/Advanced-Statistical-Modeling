import numpy as np


def Delta_func(X):
	'''
	X: An array of n data points
	return X: D matrix, numpy array of size (n-2)*n
	'''
	h = X[1:] - X[:-1] # Since Xi+1 is defined for i = 1, ..., n-1 (so X2,..,Xn), then we need to have Xi defined by a relative shift to the left by 1 index
	h1, h2 = h[1:], h[:-1] # Same for h
	vectors = [1 / h1, -(1 / h1 + 1 / h2), 1 / h2] # Predefine our values for the matrix

	n_vectors, vec_len = len(vectors), len(vectors[0])
	stacked_matrix = np.zeros((vec_len, vec_len + n_vectors - 1))

	for i, vec in enumerate(vectors): # Loop through each vector, make a matrix, then stack them horizontally ((n-2)x(n-2) -> (n-2)x(n-1) -> (n-2)x(n))
		diag_indices = np.arange(vec_len) + i
		stacked_matrix[np.arange(vec_len), diag_indices] += vec

	return stacked_matrix
	pass


def W_func(X):
	'''
	X: An array of n data points
	X: W matrix, numpy array of size (n-2)*(n-2)
	'''
	h = X[1:] - X[:-1]
	h1, h2 = h[1:], h[:-1]
	return np.diag((h1 + h2) / 3) + np.diag(h2 / 6, k=1)[1:, 1:] + np.diag(h2 / 6, k=-1)[1:, 1:]  # diag + above_diag + below_diag
	pass


def cubic_spline_regressor(X, Y, xr, Lambda):
	'''
	X: An array of n data points
	Y: An array of n response values
	xr: an array of r data points
	Lambda: positive float
	return: cubic spline regression estimation at xr, an array of r values
	'''
	n, delta, W = len(X), Delta_func(X), W_func(X) # All of this as by the formulas from the definition of the cubic spline regressor
	inv_W = np.linalg.inv(W)
	K = delta.T @ inv_W @ delta
	S = np.linalg.inv(np.eye(n) + n * Lambda * K) @ Y
	z = inv_W @ delta @ S
	z = np.concatenate(([0], z, [0])) # Since z1=zn=0

	indexes = np.searchsorted(X, xr) # The difference between each index will always be by at least 1
	idx_less_than = np.clip(indexes, 0, n - 2) # Indexes i where X[i-1] < xr <= X[i], ensuring 0 is min and n - 2 is max
	idx_greater_than = np.clip(indexes + 1, 1, n - 1) # Indexes i where X[i-1] <= xr < X[i], ensuring 1 is min and n - 1 is max

	x1, x2 = X[idx_less_than], X[idx_greater_than] # Defining X_{i} = x1, X_{i+1} = x2
	z1, z2 = z[idx_less_than], z[idx_greater_than] # Defining z_{i} = z1, z_{i+1} = z2
	s1, s2 = S[idx_less_than], S[idx_greater_than] # Defining s_{i} = s1, s_{i+1} = s2
	h = x2 - x1
	d = s1 / h - z1 * h / 6
	c = s2 / h - z2 * h / 6

	return z2 * (xr - x1) ** 3 / (6 * h) + z1 * (x2 - xr) ** 3 / (6 * h) + c * (xr - x1) + d * (x2 - xr) # f_hat by definition
	pass
	

def MSE(f, f_hat):
	'''
	f: An array of r data points
	f_hat: An array of r data points
	return:  a discrete MSE approximation, float
	'''
	return np.mean((f - f_hat)**2)
	pass



import matplotlib.pyplot as plt

if __name__ == '__main__':

	# TODO: YOUR CODE HERE
	def smooth_function(x):
		return np.sin(2 * np.pi * x)

	def f_derivative2(x):
		return -4 * np.pi ** 2 * np.sin(2 * np.pi * x)

	def lambda_func(c, x, sigma=1):
		return c * sigma ** 2 * len(x) ** (-2 / 5) / np.linalg.norm(f_derivative2(x)) ** 2

	n, sigma = 300, 1
	X = np.linspace(1 / n, 1, n)
	noise = np.random.normal(0, sigma, n)
	Y = smooth_function(X) + noise

	c_values = [0.1, 70, 1500] # Values of c for the subplots
	fig, axs = plt.subplots(1, 3, figsize=(18, 6)) # Create a figure with 1 row and 3 columns of subplots
	fig.suptitle('Cubic Spline Regressor Estimation for Smooth Function f(x) = sin(2 * pi * x)', fontsize=16)

	for i, c in enumerate(c_values): # Generate plots for each value of c
		spline_values = cubic_spline_regressor(X, Y, X, lambda_func(c, X))
		axs[i].grid(True)
		axs[i].scatter(X, Y, label='Yj = f(Xj) + noise', color='#de7f7f')
		axs[i].plot(X, smooth_function(X), label='f(Xj)', color='#dede7f', linestyle='-', linewidth=3)
		axs[i].plot(X, spline_values, label='f_hat(Xj)', color='#7fdeaf', linestyle='--', linewidth=3)
		axs[i].set_title(f'c = {c}')
		axs[i].set_xlabel('Xj')
		axs[i].set_ylabel('Yj and f(Xj)')
		axs[i].legend()

	plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to prevent overlap
	plt.show()

	# Now for Monte-Carlo trials for the MSE
	c_values = np.linspace(1, 80, 8)
	n, B, c_max = 100, 1000, len(c_values) # Used smaller parameters for short runtime taking B=10k, n=300 will take ~1 hour of runtime

	X = np.linspace(1 / n, 1, n)
	f_x = smooth_function(X)

	samples = np.reshape(np.random.normal(0, 1, c_max * n * B), (c_max, B, n))
	samples = {f'{i}': samples[i] for i in range(samples.shape[0])}

	MSEs = []
	for key, array in samples.items():  # Iterate through c_values dicts, each is an array of samples shape (B, n) for the noise
		y_i = f_x + array # We use broadcasting, so f_x:(n,) + array:(B, n) -> y_i:(B, n)
		splines = np.apply_along_axis(lambda y: cubic_spline_regressor(X, y, X, lambda_func(c_values[int(key)], X)), 1, y_i) # Apply cubic spline to estimate all rows of y given uniform X, at points also in X
		MSEs += [np.mean(np.mean((f_x - splines) ** 2, axis=1))] # Take MSE of each row (between each sample with f(Xi), n entries) then average it across all B rows
	c_opt = c_values[np.argmin(MSEs)] # Find c value which gets min MSE

	plt.plot(c_values, MSEs) # Plotting MSE as function of c, with Oracle c value found
	plt.xlabel('c')
	plt.ylabel('MSE')
	plt.title(f'MSE as Function of c, Oracle Value c* = {np.round(c_opt, 1)} (when n={n}, B={B})')
	plt.show()
	pass

