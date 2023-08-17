import pandas as pd
import numpy as np

"""Script assumes that the following /data directory
is mounted.
"""
TNG100_RATIO_PATH = (
    "/data/baryons/baryon-power-spectra/logPkRatio/logPkRatio_TNG100.dat"
)

FASTPM704_SNAPSHOTS_PATH = "/snapshot_dir/fastpm_box704/TNG_new_n4"
FASTPM1536_SNAPSHOTS_PATH = "/snapshot_dir/fastpm_box1536/TNG"
TNG100_REDSHIFTS = np.array(
    [3.71, 3.49, 3.28, 2.90, 2.44, 2.1, 1.74, 1.41, 1.04, 0.7, 0.35, 0.18, 0.0]
)


def get_TNG100_ratio():
    return pd.read_csv(TNG100_RATIO_PATH, sep="\s+")


def get_TNG100_redshifts():
    return [f"z{redshift:.2f}".replace(".", "") for redshift in TNG100_REDSHIFTS]


def get_covariance():
    return


def interpolate_TNG100_ratio(new_redshifts):
    return


def main():
    # Interpolate TNG100 redshifts to match FASTPM redshifts.

    # Interpolate TNG100 wavenumbers to match FASTPM wavenumbers.

    # Calculate the target power spectrum (Pk_TNG100_ratio * Pk_FASTPM)

    # Calculate the covariance matrix (assumed diagonal).

    # Set up likelihood function for EGD parameters.

    # Perform sampling.
    return


if __name__ == "__main__":
    main()
