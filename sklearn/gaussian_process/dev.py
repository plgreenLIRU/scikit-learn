import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (DotProduct, ConstantKernel)
from matplotlib import pyplot as plt

kernel = (ConstantKernel(0.1, (0.01, 10.0)) *
          (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2))

# Specify Gaussian Process
gp = GaussianProcessRegressor(kernel=kernel)

# Generate data and fit GP
rng = np.random.RandomState(4)
X = rng.uniform(0, 5, 10)[:, np.newaxis]
y = np.sin((X[:, 0] - 2.5) ** 2)

y = (y - np.mean(y)) / np.std(y)

gp.fit(X, y)
GaussianProcessRegressor(alpha=1e-10, copy_X_train=True,
                         kernel=0.316**2 * DotProduct(sigma_0=1)**2,
                         n_restarts_optimizer=0, normalize_y=False,
                         optimizer='fmin_l_bfgs_b', random_state=None)

# Predictions
X_ = np.linspace(0, 5, 100)
y_mean, y_std, y_var_error = gp.predict(X_[:, np.newaxis], return_std=True)

# Print 'uncorrected' y variance
print(y_var_error)

# Plots
plt.figure()
plt.plot(X, y, 'o')
plt.plot(X_, y_mean, 'k')
plt.plot(X_, y_mean + 3 * y_std, 'k')
plt.plot(X_, y_mean - 3 * y_std, 'k')
plt.show()
