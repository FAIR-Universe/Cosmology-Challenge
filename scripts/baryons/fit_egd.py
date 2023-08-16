import argparse
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation", type=int, default=0)
    parser.add_argument(
        "--base_dir",
        type=Path,
        default="/pscratch/sd/b/bthorne/fairuniverse/hsc_dataset",
    )
    parser.add_argument("--data_dir", type=Path, default="data")
    parser.add_argument("--cosmology_file", type=Path, default="data/cosmology.txt")
    parser.add_argument("--redshifts_file", type=Path, default="data/redshifts.txt")
    return vars(parser.parse_args())


def fit_egd():
    return


def main():
    args = parse_arguments()
    fit_egd(args)
    return


if __name__ == "__main__":
    main()
