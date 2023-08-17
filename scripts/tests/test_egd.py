from wlchallenge.egd import PGD, EGD
from nbodykit.source.catalog.file import BigFileCatalog

fastpm_tng_seed = BigFileCatalog(
    "/pscratch/sd/b/bthorne/fairuniverse/hsc_dataset/fastpm_box704/TNG_new_n4/Om_0.3089_S8_0.8159_0.2494",
    dataset="1",
)
print(fastpm_tng_seed.comm.rank)

fastpm_tng_seed.attrs["Nmesh"] = 128
gamma = 1.1
beta = 1.0
disp = EGD(fastpm_tng_seed, gamma, beta)
print(disp.shape)
