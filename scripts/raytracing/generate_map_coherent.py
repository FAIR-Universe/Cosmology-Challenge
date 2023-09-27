from raytracing import LensingPlane, RayTracer, EmptyLensingPlane
from nbodykit.cosmology import Cosmology
from nbodykit.utils import GatherArray
import numpy as np
import time
import gc
from mpi4py import MPI
from pmesh.pm import ParticleMesh
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation", type=int, default=0)
    parser.add_argument("--Nmesh", type=int, default=8192)
    parser.add_argument("--base_dir", type=str, default="/snapshot_dir")
    parser.add_argument("--realization", type=int, default=None)
    parser.add_argument("--realization_start", type=int, default=0)
    parser.add_argument("--realization_end", type=int, default=100)
    parser.add_argument("--cosmology_file", type=Path, default="data/cosmology.txt")
    parser.add_argument("--redshifts_file", type=Path, default="data/redshifts.txt")
    parser.add_argument("--weight_planes", action="store_true")
    args = parser.parse_args()
    params = vars(args)

    if params["realization"] is not None:
        params["realization_start"] = 5 * params["realization"]
        params["realization_end"] = 5 * (params["realization"] + 1)

    # hyperparameters
    Omega_m = np.loadtxt(params["cosmology_file"])[params["simulation"], 0]
    sigma_8 = np.loadtxt(params["cosmology_file"])[params["simulation"], 1]
    simulation = str(params["simulation"])
    # Omega_m = 0.279
    # sigma_8 = 0.82
    # simulation = 'mock'
    base_dir = params["base_dir"]
    angle = 16 / 180 * np.pi
    Nmesh = params["Nmesh"]

    # constant
    h = 0.7
    Omega_b = 0.046
    n_s = 0.97
    comm = MPI.COMM_WORLD

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

    if params["weight_planes"]:
        weights_bin1 = np.load("weights_bin1_53.npy")
        weights_bin2 = np.load("weights_bin2_53.npy")
        weights_bin3 = np.load("weights_bin3_53.npy")
        weights_bin4 = np.load("weights_bin4_53.npy")
        weights_planes = np.array(
            (weights_bin1, weights_bin2, weights_bin3, weights_bin4)
        )
    else:
        weights_planes = None

    pm = ParticleMesh(Nmesh=[Nmesh, Nmesh], BoxSize=[angle, angle])
    grid = pm.generate_uniform_particle_grid(shift=0.5)
    x = grid[:, 0]
    y = grid[:, 1]
    del grid
    gc.collect()

    starting_index = np.array([0, 3, 6, 9, 12, 18, 24, 34])

    for realization in range(params["realization_start"], params["realization_end"]):
        # random seed
        np.random.seed(realization)

        # random realization
        plane_realization = comm.bcast(
            np.random.randint(low=0, high=18, size=8) if comm.rank == 0 else None,
            root=0,
        )
        plane_realization[0] = realization % 36
        plane_random_shift = comm.bcast(
            np.random.rand(8, 2) if comm.rank == 0 else None, root=0
        )
        if realization // 36 == 0:
            plane_random_shift[0] = 0
        elif realization // 36 == 1:
            plane_random_shift[1, 0] = 0
            plane_random_shift[1, 1] = 1
        elif realization // 36 == 2:
            plane_random_shift[1, 0] = 1
            plane_random_shift[1, 1] = 0
        elif realization // 36 == 3:
            plane_random_shift[1] = 1
        plane_flip = comm.bcast(
            np.random.randn(8, 2) > 0 if comm.rank == 0 else None, root=0
        )
        plane_transpose = comm.bcast(
            np.random.randn(8) > 0 if comm.rank == 0 else None, root=0
        )

        t = time.time()

        # Load potential plane
        Tracer = RayTracer()

        for snapshot_id in range(53):
            index = np.sum(snapshot_id >= starting_index) - 1

            if snapshot_id < 43:
                plane = LensingPlane(
                    redshift=redshift[snapshot_id],
                    Omega_m=Omega_m,
                    comoving_distance=comoving_dis[snapshot_id],
                    memory_efficient=True,
                )

                sim_path = str(
                    Path(base_dir)
                    / "LensingPlane"
                    / f"{int(simulation):03d}_plane{snapshot_id:03d}_realization{plane_realization[index]:03d}_coherent"
                )
                if comm.rank == 0:
                    print(sim_path, flush=True)

                plane.load(
                    sim_path,
                    files=["potentialk"],
                )

                # random shift, transpose and flip
                if snapshot_id < 3:
                    plane.margin = [
                        plane.potentialk.pm.BoxSize[0] / 6.0
                        + plane.potentialk.pm.BoxSize[0]
                        / 3.0
                        * plane_random_shift[index, 0],
                        plane.potentialk.pm.BoxSize[1] / 6.0
                        + plane.potentialk.pm.BoxSize[1]
                        / 3.0
                        * plane_random_shift[index, 1],
                    ]
                else:
                    plane.random_shift = plane_random_shift[index]
                plane.flip = plane_flip[index]
                plane.transpose = plane_transpose[index]
            else:
                # TODO: IA signal at empty lensing plane
                plane = EmptyLensingPlane(
                    redshift=redshift[snapshot_id],
                    Omega_m=Omega_m,
                    comoving_distance=comoving_dis[snapshot_id],
                )

            Tracer.addLens(plane)

        Tracer.reorderLenses()

        # ray tracing
        # convergence_bins, shear_bins, convergence, shear, current_positions = Tracer.shoot(np.array((x, y)), z=redshift[-1], weight_planes=weights_planes, save_intermediate=False, IA=0, pz=None, Fz=None)
        hessian_path = str(
            Path(base_dir)
            / "shearMatrix"
            / f"{int(simulation):03d}_realization{realization:03d}_Nmesh{Nmesh}_plane43_shearMatrix_coherent"
        )
        (
            # convergence_bins, ## TODO : what were these doing here?
            # shear_bins,
            convergence_bins,
            shear,
            current_positions,
        ) = Tracer.shoot(
            np.array((x, y)),
            z=redshift[-1],
            weight_planes=weights_planes,
            save_intermediate=False,
            IA=0,
            pz=None,
            Fz=None,
            save_Hessian=hessian_path,
            save_Nplane=43,
        )
        del shear, current_positions, Tracer, plane
        gc.collect()

        # convergence = GatherArray(convergence, comm, root=0)
        convergence_bins = GatherArray(convergence_bins.T, comm, root=0)
        # shear_bins = GatherArray(shear_bins.transpose(2,0,1), comm, root=1)
        gc.collect()

        # save mock maps
        if comm.rank == 0:
            kappa_path = str(
                Path(base_dir)
                / "kappa_maps"
                / f"{simulation}_realization{realization}_Nmesh{Nmesh}_plane43_empty10_convergence_coherent.npy"
            )
            np.save(
                kappa_path,
                convergence_bins.T.astype(np.float32).reshape(1, Nmesh, Nmesh),
            )  # (4, Nmesh^2)
            # np.save(base_dir + 'kappa_maps/' + simulation + '_realization%d_Nmesh%d_plane43_empty10_final_convergence_coherent.npy'%(realization,Nmesh), convergence.astype(np.float32).reshape(Nmesh, Nmesh)) # (Nmesh^2)
            print(
                "Finished realization",
                realization,
                "Time:",
                time.time() - t,
                flush=True,
            )
            print()

        # if comm.rank == 1:
        #    np.save(base_dir + 'kappa_maps/' + simulation + '_realization%d_Nmesh%d_plane43_empty10_shear_coherent.npy'%(realization,Nmesh), shear_bins.transpose(1,2,0).astype(np.float32).reshape(4, 2, Nmesh, Nmesh)) #(4,2,Nmesh^2)

        #    print('Finished realization', realization, 'Time:', time.time()-t, flush=True)
        #    print()

        # del convergence_bins, shear_bins
        gc.collect()


if __name__ == "__main__":
    main()
