from wlchallenge.egd import EGD, PGD
from nbodykit.source.catalog.file import BigFileCatalog
from nbodykit.lab import FFTPower
import matplotlib.pyplot as plt

fastpm_tng_seed = BigFileCatalog(
    "/pscratch/sd/b/bthorne/fairuniverse/hsc_dataset/fastpm_box704/TNG_new_n4/Om_0.3089_S8_0.8159_0.2494",
    dataset="1",
)

fastpm_tng_seed.attrs["Nmesh"] = 1024
gamma = 1.1
beta = 1.0
alpha = 0.004
kl = 1.45
ks = 11.0

Nmesh = 1024
Pk_fastpm = FFTPower(fastpm_tng_seed, mode="1d", Nmesh=Nmesh).power

egd_adjustment = EGD(fastpm_tng_seed, gamma, beta)
fastpm_tng_seed["Position"] += egd_adjustment

Pk_egd_adjusted = FFTPower(fastpm_tng_seed, mode="1d", Nmesh=Nmesh).power

if fastpm_tng_seed.comm.rank == 0:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.semilogx(
        Pk_fastpm["k"],
        (Pk_fastpm["power"].real - Pk_fastpm.attrs["shotnoise"])
        / (Pk_fastpm["power"].real - Pk_fastpm.attrs["shotnoise"]),
        label="FastPM",
        linestyle="--",
    )
    ax.semilogx(
        Pk_egd_adjusted["k"],
        (Pk_egd_adjusted["power"].real - Pk_egd_adjusted.attrs["shotnoise"])
        / (Pk_fastpm["power"].real - Pk_fastpm.attrs["shotnoise"]),
        label="EGD Adjusted",
        linestyle=":",
    )
    ax.legend(loc=3, frameon=False)
    ax.set_xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    ax.set_ylabel(r"$P(k)/ P^{\rm FastPM}(k)$")
    ax.set_xlim(0.01, 50)
    ax.set_ylim(0.2, 2)
    fig.savefig("/plots/spectrum_egd_test.png", bbox_inches="tight")

    print(egd_adjustment)
