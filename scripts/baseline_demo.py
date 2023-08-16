import numpy as np
import h5py

# Set parameters
size = 512  # Size of the map
pixel_scale = 0.1  # pixel scale in degrees
num_cosmologies = 100  # number of cosmologies
num_realizations = 100  # number of realizations per cosmology
np.random.seed(42)
omega_m_values = np.random.uniform(0.25, 0.35, num_cosmologies)  # Omega_m values
sigma_8_values = np.random.uniform(0.7, 0.9, num_cosmologies)  # sigma_8 values


def power_spectrum(k, omega_m, sigmas_8):
    # Simple power spectrum model for demonstration
    amplitude = sigma_8**2
    return amplitude * (k / 0.1) ** (-3) * omega_m


def generate_map(omega_m, sigma_8):
    # Generate a grid of wave numbers
    kx = np.fft.fftfreq(size, d=pixel_scale) * 2 * np.pi
    ky = np.fft.fftfreq(size, d=pixel_scale) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky)
    k = np.sqrt(kx**2 + ky**2)
    k[k == 0] = 1e-4
    # Generate a realization of the convergence in Fourier space
    ft_map = np.random.normal(scale=np.sqrt(power_spectrum(k, omega_m, sigma_8)))
    # Transform to real space
    map = np.fft.ifft2(ft_map).real
    return map


# Create an HDF5 file to store the data
with h5py.File(
    "/pscratch/sd/b/bthorne/fairuniverse/demo/convergence_maps.h5", "w"
) as f:
    for i, (omega_m, sigma_8) in enumerate(zip(omega_m_values, sigma_8_values)):
        cosmology_group = f.create_group(f"cosmology_{i}")
        cosmology_group.attrs["omega_m"] = omega_m
        cosmology_group.attrs["sigma_8"] = sigma_8
        # Generate and save each realization for this cosmology
        for j in range(num_realizations):
            convergence_map = generate_map(omega_m, sigma_8)
            cosmology_group.create_dataset(f"realization_{j}", data=convergence_map)

print("Data generation complete.")
