from raytracing import LensingPlane, RayTracer
from nbodykit.cosmology import Cosmology
import numpy as np
from pmesh.pm import ParticleMesh
import time
import gc
from nbodykit.lab import FieldMesh
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation", type=int, default=0)
    parser.add_argument(
        "--base_dir",
        type=Path,
        default="/pscratch/sd/b/bthorne/fairuniverse/hsc_dataset/potentials",
    )
    parser.add_argument("--snapshot_start", type=int, default=0)
    parser.add_argument("--snapshot_end", type=int, default=43)
    parser.add_argument("--data_dir", type=Path, default="data")
    parser.add_argument("--cosmology_file", type=Path, default="data/cosmology.txt")
    args = parser.parse_args()
    params = vars(args)

    # hyperparameters
    Omega_m = np.loadtxt(params["cosmology_file"])[params["simulation"], 0]
    sigma_8 = np.loadtxt(params["cosmology_file"])[params["simulation"], 1]
    simulation = str(params["simulation"])
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

    redshift = np.array(
        [
            0.01674168,
            0.0506174,
            0.08504611,
            0.12006466,
            0.15571175,
            0.19202803,
            0.22905622,
            0.26684124,
            0.30543036,
            0.34487332,
            0.38522251,
            0.42653314,
            0.46886342,
            0.51227478,
            0.55683209,
            0.60260389,
            0.64966266,
            0.69808513,
            0.74795256,
            0.79935109,
            0.85237212,
            0.90711273,
            0.96367608,
            1.02217196,
            1.08271726,
            1.1454366,
            1.21046296,
            1.27793838,
            1.34801472,
            1.42085458,
            1.49663214,
            1.57553433,
            1.65776187,
            1.74353062,
            1.83307297,
            1.92663943,
            2.02450038,
            2.12694802,
            2.23429857,
            2.34689472,
            2.46510834,
            2.58934359,
            2.72004039,
            2.85767826,
            3.00278084,
            3.15592076,
            3.31772541,
            3.48888333,
            3.67015156,
            3.86236406,
            4.06644131,
            4.2834014,
            4.51437274,
        ]
    )

    comoving_dis = np.array([cosmology.comoving_distance(z) for z in redshift])

    snapshot_basedir = params["base_dir"]
    snapshot_dir = []
    for i in range(12):
        snapshot_dir.append(
            snapshot_basedir / "MP-Gadget" / simulation / f"PART_{15 - i:03d}"
        )
    for i in range(12, 24):
        snapshot_dir.append(
            snapshot_basedir
            / "fastpm_box704"
            / "{simulation}_new_n4"
            / f"Om_{Omega_m:.4f}_S8_{sigma_8:.4f}_{1/(1+redshift[i]):.4f}"
        )
    for i in range(24, 43):
        snapshot_dir.append(
            snapshot_basedir
            / "fastpm_box1536"
            / simulation
            / f"Om_{Omega_m:.4f}_S8_{sigma_8:.4f}_{1/(1+redshift[i]):.4f}"
        )

    starting_index = np.array([0, 3, 6, 9, 12, 18, 24, 34])
    gap = 0

    assert params["snapshot_start"] in starting_index
    for snapshot_id in range(params["snapshot_start"], params["snapshot_end"]):
        index = np.sum(snapshot_id >= starting_index) - 1
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
            periodic = False
            Nrealization = 36
        else:
            width = boxsize
            periodic = True
            Nrealization = 18

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
                part_dir=snapshot_dir[snapshot_id],
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
                save_dir
                + "LensingPlane/"
                + simulation
                + "_plane%d_realization%d_coherent" % (snapshot_id, realization),
                files=["potentialk"],
            )

            gc.collect()

        del pos, plane
        gc.collect()

        gap += thick


if __name__ == "__main__":
    main()
