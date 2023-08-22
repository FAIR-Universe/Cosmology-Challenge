import os


def get_subdirectories(root_dir):
    return [
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]


def get_FastPM_scales_from_directories(dir):
    paths = get_subdirectories(dir)
    # Snapshot directories saved with scale factor to 4 decimal places at
    # end of path.
    return [float(path[-6:]) for path in paths]


def get_FastPM_redshifts_from_directories(dir):
    return list(map(scale_to_redshift, get_FastPM_scales_from_directories(dir)))


def redshift_to_scale(z):
    return 1.0 / (1.0 + z)


def scale_to_redshift(a):
    return 1.0 / a - 1.0
