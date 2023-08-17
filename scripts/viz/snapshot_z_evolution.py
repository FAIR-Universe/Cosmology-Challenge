from nbodykit.source.catalog.file import BigFileCatalog
import matplotlib.pyplot as plt
import numpy as np

# These are the wrong redshifts, actually. Need the ones that Biwei ran with
# but I don't have access to those.
REDSHIFTS = np.load("/data/baryons/redshifts.txt")

CAT_FILES = [
    f"/snapshot_dir/fastpm_box704/TNG_new_n4/Om_0.3089_S8_0.8159_{1 / (1 + z):.4f}"
    for z in REDSHIFTS
]


nrows = 4
ncols = 4
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
