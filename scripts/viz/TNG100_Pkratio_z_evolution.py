import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.pyplot as plt

TNG100_RATIO_PATH = (
    "/data/baryons/baryon-power-spectra/logPkRatio/logPkRatio_TNG100.dat"
)
TNG100 = pd.read_csv(TNG100_RATIO_PATH, sep="\s+")

PLOTTING_DIR = "/plots"

# TNG redshifts copied from baryon-power-spectra
REDSHIFTS = np.array(
    [3.71, 3.49, 3.28, 2.90, 2.44, 2.1, 1.74, 1.41, 1.04, 0.7, 0.35, 0.18, 0.0]
)


def main():
    fig = plt.figure(figsize=(7.2, 6.5))
    plt.rc("font", size=20)
    title = r"${\rm TNG}100~P^{\rm hydro}(k) / P^{\rm DMO}(k)$"

    ax1 = fig.add_axes([0.158, 0.11, 0.813, 0.84])

    colormap = cm.get_cmap("viridis")
    norm = plt.Normalize(np.min(REDSHIFTS), np.max(REDSHIFTS))

    for z in REDSHIFTS:
        ax1.plot(
            10 ** TNG100["logk"],
            10 ** TNG100[f"z{z:.2f}".replace(".", "")] - 1,
            color=colormap(norm(z)),
            lw=3,
            ls="-.",
        )

    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])  # This line is required but the array is not used.

    # Add the colorbar, including a label if desired
    plt.colorbar(sm, ax=ax1, orientation="vertical", label="Redshift")

    ax1.axhline(y=0, color="gray", linestyle="-", label=r"$\mathrm{DMO}$", lw=2)
    ax1.axvline(x=10, color="k", linestyle="--")

    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\mathrm{k\ [Mpc^{-1}h]}$")
    ax1.set_ylabel(r"$\mathrm{P_{\delta}^{hydro}(k)/P_{\delta}^{DMO}(k)-1}$")
    ax1.set_title(title)

    ax1.set_xlim(0.05, 100)

    ax1.legend(loc="best", prop={"size": 14.5}, ncol=2, frameon=False)
    ax1.set_ylim(-0.4, 0.9)

    fig.savefig(Path(PLOTTING_DIR) / "logPkRatio_TNG100.png", bbox_inches="tight")
    fig.savefig(Path(PLOTTING_DIR) / "logPkRatio_TNG100.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
