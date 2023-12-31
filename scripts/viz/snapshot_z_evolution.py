from nbodykit.source.catalog.file import BigFileCatalog
import matplotlib.pyplot as plt
import numpy as np
import os
from mpi4py import MPI
import time
import warnings
import argparse
from pathlib import Path

warnings.filterwarnings("ignore")

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

print("Reporting from rank {}".format(RANK))

REDSHIFTS = np.loadtxt("/data/baryons/redshifts.txt")

SNAPSHOT_DIR = "/snapshot_dir/fastpm_box704/TNG_new_n4"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    params = vars(args)
    return params


def get_subdirectories(root_dir):
    return [
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]


def read_mesh_and_paint(path, Nmesh=256):
    snapshot = BigFileCatalog(path, dataset="1", comm=COMM)
    mesh = snapshot.to_mesh(Nmesh=Nmesh, resampler="tsc")
    return mesh.paint(mode="real").preview(axes=[1, 2], Nmesh=Nmesh)


if RANK == 0:
    print("Calculating deltas ...")

subdirectories = get_subdirectories(SNAPSHOT_DIR)
total_snapshots = len(subdirectories)
one_plus_deltas = []

for idx, path in enumerate(subdirectories):
    if RANK == 0:
        start_time = time.time()
        print(f"Processing snapshot {idx + 1}/{total_snapshots}...")
    delta = read_mesh_and_paint(path)
    one_plus_deltas.append(delta)

    if RANK == 0:
        end_time = time.time()
        duration = end_time - start_time
        # Print a progress update from each rank, including the duration of the iteration
        print(
            f"Completed snapshot {idx + 1}/{total_snapshots} in {duration:.2f} seconds with {SIZE:d} processes.s"
        )


if RANK == 0:
    params = parse_args()
    nrows = 4
    ncols = 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
    axes = axes.flatten()
    for one_plus_delta, ax in zip(one_plus_deltas, axes):
        ax.imshow(np.log10(one_plus_delta))
    axes[-1].imshow(one_plus_deltas[-1] - one_plus_deltas[0])
    fig.savefig(params["output"], bbox_inches="tight")
