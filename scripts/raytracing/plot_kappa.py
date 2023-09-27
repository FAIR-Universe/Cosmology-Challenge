import numpy as np
import matplotlib.pyplot as plt

fpath = "/snapshot_dir/kappa_maps/0_realization0_Nmesh8192_plane43_empty10_convergence_coherent.npy"
convergence = np.load(fpath)

fig, ax = plt.subplots(1, 1)
ax.imshow(convergence[0], vmin=-0.1, vmax=0.1, cmap="seismic")
ax.set_title("$\kappa$")
fig.savefig("/plots/convergence.pdf", bbox_inches="tight")
