from nbodykit.source.catalog.file import BigFileCatalog
import numpy as np

files = [
    (
        "/pscratch/sd/b/biwei/HSC/fastpm_box704/TNG_new_n4/Om_0.3089_S8_0.8159_0.2494",
        2816,
        1,
    ),
    ("/pscratch/sd/b/biwei/HSC/fastpm_box1536/TNG/Om_0.3089_S8_0.8159_0.4149", 1536, 1),
    (
        "/pscratch/sd/b/bthorne/fairuniverse/hsc_dataset/fastpm_box704/0_new_n4/Om_0.3000_S8_0.8000_0.6808",
        2816,
        0.4,
    ),
    (
        "/pscratch/sd/b/bthorne/fairuniverse/hsc_dataset/fastpm_box1536/0/Om_0.3000_S8_0.8000_0.2688",
        1536,
        0.4,
    ),
]

for file, nc, frac_saved in files:
    bfc = BigFileCatalog(file, dataset="1")
    print(file)
    print(
        f"\n Number of particles: {len(bfc):.3e}, \n Expected ({nc})^3 * {frac_saved}: ",
        f"{nc**3 * frac_saved:.3e} \n",
    )
