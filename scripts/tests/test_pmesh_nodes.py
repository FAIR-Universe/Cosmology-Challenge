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

NMESH = 1024

print("Reporting from rank {}".format(RANK))

SNAPSHOT_DIR = "/snapshot_dir/fastpm_box704/TNG_new_n4"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--title", type=str)
    args = parser.parse_args()
    params = vars(args)
    return params


def get_subdirectories(root_dir):
    return [
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]


def read_mesh_and_paint(path):
    snapshot = BigFileCatalog(path, dataset="1", comm=COMM)
    mesh = snapshot.to_mesh(Nmesh=NMESH, resampler="tsc")
    return (path, mesh.paint(mode="real").preview(axes=[1, 2], Nmesh=NMESH))


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
    fig.suptitle(f"Nodes: {params['title']}")
    axes = axes.flatten()
    for one_plus_delta, ax in zip(one_plus_deltas, axes):
        (path, img) = one_plus_delta
        ax.imshow(np.log10(img))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_title(path.split("/")[-1])
        fig.savefig(f"{params['output']}_Nmesh{NMESH}.pdf", bbox_inches="tight")
        fig.savefig(f"{params['output']}_Nmesh{NMESH}.png", bbox_inches="tight")
