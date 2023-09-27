import os


def get_subdirectories(root_dir):
    return [
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]


def get_FastPM_scales_from_directories(dir):
    paths = get_subdirectories(dir)
    # Snapshot directories saved with scale factor to 4 decimal places at
    # end of path.
    return [float(path[-6:]) for path in paths]


def get_FastPM_redshifts_from_directories(dir):
    return list(map(scale_to_redshift, get_FastPM_scales_from_directories(dir)))


def redshift_to_scale(z):
    return 1.0 / (1.0 + z)


def scale_to_redshift(a):
    return 1.0 / a - 1.0


def get_covariance_matrix(ks, Pk):
    """Calculate diagonal covariance matrix for power spectrum.

    We assume that the covariance matrix is diagonal, so we only need to
    calculate the variance of each power spectrum bin. The variance
    is assumed to be given by:

    .. math::
        \mathrm{Var}(P(k)) = (0.1 * P(k)) ^ 2

    As this was found to give empirically good results in Dai et al 2018.

    Args:
        ks (ndarray): Wavenumbers in h/Mpc.
        Pk (ndarray): Power in corresponding bins.
    """
    if np.isnan(Pk).any():
        # NaNs caused by too-low Nmesh can result in high-k NaNs.
        raise ValueError("Pk contains NaNs, try increasing Nmesh")
    return (0.1 * np.diag(Pk)) ** 2