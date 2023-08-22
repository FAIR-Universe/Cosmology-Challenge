import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpi4py import MPI

import os

from nbodykit.source.catalog.file import BigFileCatalog
from nbodykit.lab import FFTPower
from wlchallenge.egd import EGD
import emcee
import corner

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

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
TNG100_REDSHIFT_COLUMNS = [
    f"z{redshift:.2f}".replace(".", "") for redshift in TNG100_REDSHIFTS
]


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


def get_TNG100_ratio():
    return pd.read_csv(TNG100_RATIO_PATH, sep="\s+")


def get_TNG100_redshifts():
    return [f"z{redshift:.2f}".replace(".", "") for redshift in TNG100_REDSHIFTS]


def interpolate_TNG100_redshifts(df, new_redshifts):
    # Interpolating for each row (power spectrum at a given scale) across z
    for new_redshift in new_redshifts:
        new_column = []
        for _, row in df.iterrows():
            power_values = row[TNG100_REDSHIFT_COLUMNS].values
            interpolator = interp1d(TNG100_REDSHIFTS, power_values, kind="linear")
            interpolated_value = interpolator(new_redshift)
            new_column.append(interpolated_value)
        df[f"z{new_redshift:.2f}".replace(".", "")] = new_column
    return df


def interpolate_TNG100_Pkratio(df, redshift_column, wavenumbers):
    # Extract the log wavenumbers and power for the specified redshift
    logk_values = df["logk"].values
    log_power_values = df[redshift_column].values

    # We need to convert logk_values and log power values back to
    # k_values and ratios for interpolation
    k_values = 10**logk_values
    power_values = 10**log_power_values
    interpolator = interp1d(
        k_values, power_values, kind="cubic", fill_value="extrapolate"
    )

    # Use the interpolating function to calculate the power at the specified
    # wavenumbers
    interpolated_power = interpolator(wavenumbers)

    return interpolated_power


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
    return (0.1 * np.diag(Pk)) ** 2


def plot_interpolated_Pk_ratio(df, Pkratio, colz, ks):
    title = f"Interpolated ratio at {colz}"
    fig = plt.figure(figsize=(7.2, 6.5))
    plt.rc("font", size=20)

    ax1 = fig.add_axes([0.158, 0.11, 0.813, 0.84])

    ax1.plot(
        10 ** df["logk"],
        10 ** df[colz] - 1,
        label=r"$\mathrm{TNG100}$",
        color="lightskyblue",
        lw=3,
        ls="-.",
    )
    ax1.plot(
        ks,
        Pkratio - 1,
        label=r"$\mathrm{TNG100}~{}\rm interp$",
        color="green",
        lw=3,
        ls=":",
    )

    ax1.axhline(y=0, color="gray", linestyle="-", label=r"$\mathrm{DMO}$", lw=4)
    ax1.axvline(x=30, color="k", linestyle="--")

    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\mathrm{k\ [Mpc^{-1}h]}$")
    ax1.set_ylabel(r"$\mathrm{P_{\delta}^{hydro}(k)/P_{\delta}^{DMO}(k)-1}$")
    ax1.set_title(title)

    ax1.set_xlim(0.05, 5)

    ax1.legend(loc="best", prop={"size": 14.5}, ncol=2, frameon=False)
    ax1.set_ylim(-0.4, 0.9)
    print(f"Saving to /plots/baryons/Pkratio_interp_{colz}.pdf")
    fig.savefig(f"/plots/baryons/Pkratio_interp_{colz}.pdf", bbox_inches="tight")


def plot_PkFastPM(Pk, target_Pk, cov_Pk):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    sigma_Pk = np.sqrt(np.diag(cov_Pk))

    ax.loglog(
        Pk["k"],
        Pk["power"].real - Pk.attrs["shotnoise"],
        label="FastPM 704",
        color="k",
        linestyle="-",
    )

    lower = Pk["power"].real - Pk.attrs["shotnoise"] - sigma_Pk
    upper = Pk["power"].real - Pk.attrs["shotnoise"] + sigma_Pk
    ax.fill_between(Pk["k"], lower, upper, color="gray", alpha=0.5)

    ax.loglog(
        Pk["k"],
        target_Pk,
        label="Target",
        color="lightskyblue",
        linestyle="--",
    )

    ax.legend(loc=3, frameon=False)
    ax.set_xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    ax.set_ylabel(r"$P(k)/ P^{\rm FastPM}(k)$")
    ax.set_xlim(0.1, 10)
    print("Saving to /plots/baryons/FastPM704_Pk.pdf")
    fig.savefig("/plots/baryons/FastPM704_Pk.pdf", bbox_inches="tight")


class FastPM_EGD_Likelihood(object):
    def __init__(self, Pk_target, cov_Pk, cat, Nmesh=1024, kmax=10):
        self.Pk_target = Pk_target
        self.cov_Pk = cov_Pk
        self.cat = cat
        self.Nmesh = Nmesh
        self.kmax = kmax
        return

    def __call__(self, params):
        if RANK == 0:
            print(f"Gamma {params[0]:.3f}, beta {params[1]:.3f}", flush=True)
        if params[0] < 1:
            return -np.inf
        pos = EGD(self.cat, params[0], params[1])
        Pk = FFTPower(pos, Nmesh=self.Nmesh, kmax=self.kmax).power
        delta_Pk = Pk["power"].real - Pk.attrs["shotnoise"] - self.Pk_target
        return -0.5 * np.dot(delta_Pk, np.linalg.solve(self.cov_Pk, delta_Pk))


def main():
    # Snapshot parameters
    a = 0.6579
    z = 1 / a - 1
    colz = f"z{z:.2f}".replace(".", "")

    # Power spectrum parameters
    Nmesh_Pk = 1024
    kmax = 10.0

    # Emcee parameters
    ndim = 2
    nwalkers = 50

    # Interpolate TNG100 redshifts to match FASTPM redshifts.
    FPM704_redshifts = get_FastPM_redshifts_from_directories(FASTPM704_SNAPSHOTS_PATH)
    df = get_TNG100_ratio()
    df = interpolate_TNG100_redshifts(df, FPM704_redshifts)

    # Interpolate TNG100 wavenumbers to match FASTPM wavenumbers.

    if RANK == 0:
        ks = np.logspace(-2, 1, 100)
        Pkratio = interpolate_TNG100_Pkratio(df, colz, ks)
        plot_interpolated_Pk_ratio(df, Pkratio, colz, ks)

    # Calculate the target power spectrum (Pk_TNG100_ratio * Pk_FASTPM)
    # Calculate Pk_FASTPM704
    cat = BigFileCatalog(
        f"{FASTPM704_SNAPSHOTS_PATH}/Om_0.3089_S8_0.8159_{a:.4f}", dataset="1"
    )
    Pk_fastpm704 = FFTPower(cat, mode="1d", Nmesh=Nmesh_Pk, kmax=kmax).power

    target_Pk = (
        Pk_fastpm704["power"].real - Pk_fastpm704.attrs["shotnoise"]
    ) * interpolate_TNG100_Pkratio(df, colz, Pk_fastpm704["k"])

    # Calculate the covariance matrix (assumed diagonal).
    cov_Pk = get_covariance_matrix(Pk_fastpm704["k"], target_Pk)

    if RANK == 0:
        plot_PkFastPM(Pk_fastpm704, target_Pk, cov_Pk)

    # Set up likelihood function for EGD parameters.
    lkl = FastPM_EGD_Likelihood(target_Pk, cov_Pk, cat, Nmesh=Nmesh_Pk, kmax=kmax)

    # Perform sampling.
    if RANK == 0:
        print("Starting sampling...")
        backend = emcee.backends.HDFBackend("/data/baryons/checkpoint.h5")
        p0 = [np.random.rand(ndim) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, lkl, backend=backend, progress=True
        )
        sampler.run_mcmc(
            p0, 10, progress=True
        )  # 1000 is the number of samples; you can adjust as needed
        samples = sampler.get_chain(
            discard=1, thin=1, flat=True
        )  # Adjust discard and thin according to your needs

        fig = corner.corner(samples, labels=["gamma", "beta"])
        fig.savefig("/plots/baryons/posterior.pdf", bbox_inches="tight")
        plt.show()
    return


if __name__ == "__main__":
    main()
