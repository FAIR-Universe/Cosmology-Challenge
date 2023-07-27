#!/pscratch/sd/b/bthorne/conda/nbodykit_env/bin/python
import matplotlib.pyplot as plt

from nbodykit.lab import FFTPower
from nbodykit.source.catalog.file import BigFileCatalog

bfc = BigFileCatalog(
    "/pscratch/sd/b/bthorne/fairuniverse/hsc_dataset/fastpm_box1536/0/Om_0.3000_S8_0.8000_0.2688",
    dataset="1",
)

Nmeshs = [512, 1024, 2048]
kmax = 50
kmin = 0.01

fig, ax = plt.subplots(figsize=(8, 6))
for Nmesh in Nmeshs:
    result = FFTPower(bfc, mode="1d", Nmesh=Nmesh, dk=0.005, kmin=kmin, kmax=kmax)
    result.save("Pk_Nmesh_{Nmesh}.json".format(Nmesh=Nmesh))
    Pk = result.power
    ax.loglog(
        Pk["k"],
        Pk["power"].real - Pk.attrs["shotnoise"],
        label=r"Nmesh={Nmesh}".format(Nmesh=Nmesh),
    )

ax.legend(loc=3, frameon=False)
ax.set_xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
ax.set_ylabel(r"$P(k)$ [$h^{-3}\mathrm{Mpc}^3$]")
ax.set_xlim(kmin, kmax)
fig.savefig("Pk_Nmeshs.png", bbox_inches="tight")
