import numpy as np
import h5py
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


def cholesky_to_array(cholesky_matrix):
    n = cholesky_matrix.shape[0]
    array = []
    for i in range(n):
        for j in range(i + 1):
            array.append(cholesky_matrix[i, j])
    return np.array(array)


def array_to_cholesky(array, n):
    cholesky_matrix = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i + 1):
            cholesky_matrix[i, j] = array[idx]
            idx += 1
    return cholesky_matrix


# Load the covariance matrices
covariance_matrices = np.load(
    "/pscratch/sd/b/bthorne/fairuniverse/demo/covariance_matrices.npy"
)
parameter_values = []
with h5py.File(
    "/pscratch/sd/b/bthorne/fairuniverse/demo/convergence_maps.h5", "r"
) as f:
    num_cosmologies = 100
    maps = []
    labels = []
    for i in range(num_cosmologies):
        cosmology_group = f[f"cosmology_{i}"]
        omega_m = cosmology_group.attrs["omega_m"]
        sigma_8 = cosmology_group.attrs["sigma_8"]
        parameter_values.append((omega_m, sigma_8))
parameter_values = np.array(parameter_values)

# Perform the Cholesky decomposition and store the lower triangular matrices
cholesky_factors = [
    cholesky_to_array(np.linalg.cholesky(cov)) for cov in covariance_matrices
]

# Flatten the cholesky factors to a shape suitable for interpolation
cholesky_arrays = np.array(cholesky_factors)

# Define the grid for Omega_m and Sigma_8
omega_m_grid = np.linspace(0.25, 0.35, 100)
sigma_8_grid = np.linspace(0.7, 0.9, 100)

# Create a meshgrid
omega_m_mesh, sigma_8_mesh = np.meshgrid(omega_m_grid, sigma_8_grid)
grid_points = np.column_stack([omega_m_mesh.ravel(), sigma_8_mesh.ravel()])

# Create a container for the interpolated Cholesky arrays
interpolated_cholesky_arrays = []

# Iterate over each element in the Cholesky arrays
# print(grid_points.shape)
# print(parameter_values.shape)
# print(cholesky_arrays.shape)
# print(cholesky_arrays)

for i in range(cholesky_arrays.shape[1]):
    cholesky_values = cholesky_arrays[:, i]
    interpolated_values = griddata(
        parameter_values, cholesky_values, grid_points, method="nearest"
    )
    print(interpolated_values)
    interpolated_cholesky_arrays.append(interpolated_values.reshape(omega_m_mesh.shape))

interpolated_cholesky_arrays = np.array(interpolated_cholesky_arrays)
print(interpolated_cholesky_arrays.shape)

determinants = np.empty((100, 100))
exponent = np.empty((100, 100))
theta_obs = np.array([0.3, 0.8])

for i in range(100):
    for j in range(100):
        L = array_to_cholesky(interpolated_cholesky_arrays[:, i, j], 2)
        C = L @ L.T
        determinants[i, j] = 1 / np.sqrt(np.linalg.det(C))
        delta_theta = np.array([omega_m_mesh[i, j], sigma_8_mesh[i, j]]) - theta_obs
        exponent[i, j] = -0.5 * delta_theta.T @ np.linalg.inv(C) @ delta_theta


lkl = np.exp(exponent)
fig, ax = plt.subplots(1, 1)
ax.contourf(omega_m_mesh, sigma_8_mesh, lkl, levels=3, cmap="viridis")
fig.savefig("determinants.png")
