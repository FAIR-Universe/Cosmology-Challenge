"""This script tests the MPI installation. 

Sometimes when installing MPI something can get screwed
up, and all processes will be assigned rank 0. This 
just tests that that is not happening.
"""

from mpi4py import MPI


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print(f"Hello from process {rank} out of {size}")


if __name__ == "__main__":
    main()
