from raytracing import LensingPlane
from nbodykit.cosmology import Cosmology
import numpy as np
import gc
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation", type=int, default=0)
    parser.add_argument(
        "--base_dir",
        type=Path,
        default="/pscratch/sd/b/bthorne/fairuniverse/hsc_dataset",
    )
    parser.add_argument("--snapshot_start", type=int, default=0)
    parser.add_argument("--snapshot_end", type=int, default=43)
    parser.add_argument("--data_dir", type=Path, default="data")
    parser.add_argument("--cosmology_file", type=Path, default="data/cosmology.txt")
    parser.add_argument("--redshifts_file", type=Path, default="data/redshifts.txt")
    args = parser.parse_args()
    params = vars(args)

    # hyperparameters
    Omega_m = np.loadtxt(params["cosmology_file"])[params["simulation"], 0]
    sigma_8 = np.loadtxt(params["cosmology_file"])[params["simulation"], 1]
    simulation = params["simulation"]
    # Omega_m = 0.279
    # sigma_8 = 0.82
    # simulation = 'mock'
    save_dir = params["base_dir"]
    angle = 16 / 180 * np.pi
    resolution = 4e-5  # angle / Nmesh

    # constant
    h = 0.7
    Omega_b = 0.046
    n_s = 0.97

    # cosmology
    cosmology = Cosmology(h=h, Omega0_b=Omega_b, Omega0_cdm=Omega_m - Omega_b, n_s=n_s)
    Omega_ncdm = (
        cosmology.Omega0_ncdm_tot - cosmology.Omega0_pncdm_tot + cosmology.Omega0_dcdm
    )
    cosmology = Cosmology(
        h=h, Omega0_b=Omega_b, Omega0_cdm=Omega_m - Omega_ncdm - Omega_b, n_s=n_s
    )
    # cosmology = cosmology.match(sigma8=sigma_8)

    redshift = np.loadtxt(params["redshifts_file"])

    comoving_dis = np.array([cosmology.comoving_distance(z) for z in redshift])

    snapshot_basedir = params["base_dir"]
    snapshot_dir = []
    for i in range(12):
        snapshot_dir.append(
            snapshot_basedir / "MP-Gadget" / f"{simulation}" / f"PART_{15 - i:03d}"
        )
    for i in range(12, 24):
        snapshot_dir.append(
            snapshot_basedir
            / "fastpm_box704"
            / f"{simulation}_new_n4"
            / f"Om_{Omega_m:.4f}_S8_{sigma_8:.4f}_{1/(1+redshift[i]):.4f}"
        )
    for i in range(24, 43):
        snapshot_dir.append(
            snapshot_basedir
            / "fastpm_box1536"
            / f"{simulation}"
            / f"Om_{Omega_m:.4f}_S8_{sigma_8:.4f}_{1/(1+redshift[i]):.4f}"
        )

    starting_index = np.array([0, 3, 6, 9, 12, 18, 24, 34])
    gap = 0

    assert params["snapshot_start"] in starting_index
    for snapshot_id in range(params["snapshot_start"], params["snapshot_end"]):
        index = (
            np.sum(snapshot_id >= starting_index) - 1
        )  # TODO : What does this line do? Very confusing.
        if snapshot_id == starting_index[index]:
            gap = 0

        shift = np.load(params["data_dir"] / "shift_coherent.npy")[:, index]
        dim = np.load(params["data_dir"] / "dim_coherent.npy")[:, index]
        sign = np.load(params["data_dir"] / "sign_coherent.npy")[:, index]

        # load snapshot
        if snapshot_id < 12:
            boxsize = 320
            part_in_kpc = True
        elif snapshot_id < 24:
            boxsize = 704
            part_in_kpc = False
        else:
            boxsize = 1536
            part_in_kpc = False

        # FOV and thickness
        if snapshot_id < 3:
            width = comoving_dis[snapshot_id] * angle * 3
            # periodic = False
            # Nrealization = 36
            Nrealization = 1
        else:
            width = boxsize
            # periodic = True
            # Nrealization = 18
            Nrealization = 1

        Nmesh = 2 ** round(np.log2(width / comoving_dis[snapshot_id] / resolution))
        if snapshot_id > 32:
            Nmesh = min(Nmesh, 8192)

        if snapshot_id == 0:
            thick = (comoving_dis[0] + comoving_dis[1]) / 2.0
        else:
            thick = (
                comoving_dis[snapshot_id + 1] - comoving_dis[snapshot_id - 1]
            ) / 2.0

        pos = None

        for realization in range(Nrealization):
            shift[realization, dim[realization]] += sign[realization] * gap

            plane = LensingPlane(
                redshift=redshift[snapshot_id],
                Omega_m=Omega_m,
                comoving_distance=comoving_dis[snapshot_id],
                memory_efficient=True,
            )

            pos = plane.calculate_potential(
                part_dir=str(snapshot_dir[snapshot_id]),
                Nmesh=Nmesh,
                boxsize=[width, width, thick],
                los=dim[realization],
                part_in_kpc=part_in_kpc,
                shift=shift[realization],
                pos=pos,
                return_pos=True,
                verbose=True,
            )

            plane.save(
                str(
                    save_dir
                    / "LensingPlane"
                    / f"{simulation:03d}_plane{snapshot_id:03d}_realization{realization:03d}_coherent"
                ),
                files=["potentialk"],
            )

            gc.collect()

        del pos, plane
        gc.collect()

        gap += thick


if __name__ == "__main__":
    main()
