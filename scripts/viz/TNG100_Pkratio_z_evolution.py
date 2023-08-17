import pandas as pd
import numpy as np
import matplotlib.cm as cm

import matplotlib.pyplot as plt

baryon_power_spectra_path = "/global/homes/b/bthorne/projects/berkeley/FAIR-Universe/baryon-power-spectra/logPkRatio/logPkRatio_TNG100.dat"
Tb_TNG100 = pd.read_csv(baryon_power_spectra_path, sep="\s+")

# TNG redshifts copied from baryon-power-spectra
redshifts = np.array(
    [3.71, 3.49, 3.28, 2.90, 2.44, 2.1, 1.74, 1.41, 1.04, 0.7, 0.35, 0.18, 0.0]
)
redshift_columns = [f"z{redshift:.2f}".replace(".", "") for redshift in redshifts]

title = r"${\rm TNG}100~P^{\rm hydro}(k) / P^{\rm DMO}(k)$"


fig = plt.figure(figsize=(7.2, 6.5))
plt.rc("font", size=20)

ax1 = fig.add_axes([0.158, 0.11, 0.813, 0.84])

colormap = cm.get_cmap("viridis")
norm = plt.Normalize(np.min(redshifts), np.max(redshifts))

for z, colz in zip(redshifts, redshift_columns):
    ax1.plot(
        10 ** Tb_TNG100["logk"],
        10 ** Tb_TNG100[colz] - 1,
        color=colormap(norm(z)),
        lw=3,
        ls="-.",
    )

sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])  # This line is required but the array is not used.

# Add the colorbar, including a label if desired
cbar = plt.colorbar(sm, ax=ax1, orientation="vertical", label="Redshift")

ax1.axhline(y=0, color="gray", linestyle="-", label=r"$\mathrm{DMO}$", lw=2)
ax1.axvline(x=10, color="k", linestyle="--")

ax1.set_xscale("log")
ax1.set_xlabel(r"$\mathrm{k\ [Mpc^{-1}h]}$")
ax1.set_ylabel(r"$\mathrm{P_{\delta}^{hydro}(k)/P_{\delta}^{DMO}(k)-1}$")
ax1.set_title(title)

ax1.set_xlim(0.05, 100)

ax1.legend(loc="best", prop={"size": 14.5}, ncol=2, frameon=False)
ax1.set_ylim(-0.4, 0.9)

fig.savefig("plots/logPkRatio_TNG100.png", bbox_inches="tight")
fig.savefig("plots/logPkRatio_TNG100.pdf", bbox_inches="tight")
