# rewrite lenstools.simulations.raytracing with nbodykit for MPI parallelization

import time
import gc
import numpy as np
from nbodykit.source.catalog.file import BigFileCatalog
from nbodykit.lab import FieldMesh
from nbodykit.source.mesh import BigFileMesh
from nbodykit.utils import GatherArray
from pmesh.pm import ParticleMesh
from mpi4py import MPI
import pickle
import os

# Enable garbage collection if not active already
if not gc.isenabled():
    gc.enable()


###########################################################
#################fourier kernels###########################
###########################################################


def laplace(k, v):
    kk = sum(ki**2 for ki in k)
    mask = (kk == 0).nonzero()
    kk[mask] = 1
    b = v / kk
    b[mask] = 0
    return b


def gradient(dir, order=1):
    if order == 0:

        def kernel(k, v):
            # clear the nyquist to ensure field is real
            mask = v.i[dir] != v.Nmesh[dir] // 2
            return v * (1j * k[dir]) * mask

    if order == 1:

        def kernel(k, v):
            cellsize = v.BoxSize[dir] / v.Nmesh[dir]
            w = k[dir] * cellsize

            a = 1 / (6.0 * cellsize) * (8 * np.sin(w) - np.sin(2 * w))
            # a is already zero at the nyquist to ensure field is real
            return v * (1j * a)

    return kernel


def hessian(index=[0, 0]):
    def kernel(k, v):
        return -v * k[index[0]] * k[index[1]]

    return kernel


def CompensateWindow(p):
    """
    Return the Fourier-space kernel that accounts for the convolution of
    the gridded field with the window function in configuration space.

    .. note::
        see equation 18 of
        `Jing et al 2005 <https://arxiv.org/abs/astro-ph/0409240>`_

        https://github.com/bccp/nbodykit/blob/master/nbodykit/source/mesh/catalog.py

    """

    def kernel(k, v):
        kN = [np.pi * v.Nmesh[i] / v.BoxSize[i] for i in range(len(k))]
        for i in range(len(k)):
            v = v / np.sinc(k[i] / 2 / kN[i]) ** p
        return v

    return kernel


###########################################################
#################Plane class###############################
###########################################################


class LensingPlane:
    def __init__(
        self,
        redshift,
        Omega_m,
        comoving_distance,
        fac=None,
        periodic=True,
        margin=[0, 0],
        resampler="tsc",
        memory_efficient=True,
        comm=MPI.COMM_WORLD,
    ):
        # fac: 1.5 * H0^2 / c^2 * Omega_m * comoving_dis * delta_covoming_dis / scale_fac
        # The definition of x and y are opposite compared to lenstools Plane. Equivalent to lenstools plane tranpose.

        self.redshift = redshift
        self.Omega_m = Omega_m
        self.comoving_distance = comoving_distance
        self.fac = fac
        self.periodic = periodic
        self.margin = margin
        self.resampler = resampler
        self.memory_efficient = memory_efficient
        self.comm = comm

        self.random_shift = [0, 0]
        self.flip = [False, False]
        self.transpose = False
        self.potentialk = None
        self.deflection_x = None
        self.deflection_y = None
        self.shear_xx = None
        self.shear_xy = None
        self.shear_yy = None

    def calculate_potential(
        self,
        part_dir,
        Nmesh,
        boxsize,
        los=2,
        part_in_kpc=False,
        shift=[0, 0, 0],
        flip=(False, False),
        transpose=False,
        pos=None,
        return_pos=False,
        verbose=True,
    ):
        # self.comm.Barrier()
        t = time.time()

        # load particles
        part = BigFileCatalog(part_dir, dataset="1/", header="Header", comm=self.comm)
        if pos is None:
            pos = part["Position"].compute()
        Ntotal = part._source.size
        part_boxsize = part.attrs["BoxSize"][0]

        if part_in_kpc:
            pos /= 1000.0
            part_boxsize /= 1000.0

        # self.comm.Barrier()
        if verbose and self.comm.rank == 0:
            print(
                "Loading particle at redshift %.2f takes" % self.redshift,
                time.time() - t,
                "s",
                flush=True,
            )

        t = time.time()

        # boxsize
        assert boxsize[2] <= part_boxsize

        self.periodic = True
        if boxsize[0] >= part_boxsize:
            boxsize[0] = part_boxsize
        else:
            self.periodic = False
            self.random_shift = [0, 0]

        if boxsize[1] >= part_boxsize:
            boxsize[1] = part_boxsize
        else:
            self.periodic = False
            self.random_shift = [0, 0]

        if not self.periodic:
            self.margin[0] = boxsize[0] / self.comoving_distance / 4.0
            self.margin[1] = boxsize[1] / self.comoving_distance / 4.0

        # shift
        pos += shift
        pos %= part_boxsize

        # random rotation
        if los == 0:
            x, y = 1, 2
        elif los == 1:
            x, y = 0, 2
        elif los == 2:
            x, y = 0, 1

        # transpose
        if transpose:
            x, y = y, x

        # select particles in the FOV (center of the box)
        pos[:, x] -= part_boxsize / 2 - boxsize[0] / 2
        pos[:, y] -= part_boxsize / 2 - boxsize[0] / 2
        pos %= part_boxsize
        if return_pos:
            original_pos = pos
        select = (
            (pos[:, x] < boxsize[0])
            & (pos[:, y] < boxsize[1])
            & (pos[:, los] < boxsize[2])
        )
        pos = pos[select][:, [x, y]]

        if return_pos:
            original_pos[:, x] += part_boxsize / 2 - boxsize[0] / 2
            original_pos[:, y] += part_boxsize / 2 - boxsize[0] / 2
            original_pos -= shift
            original_pos %= part_boxsize
            if part_in_kpc:
                original_pos *= 1000.0

        # random reflection
        if flip[0]:
            pos[:, 0] = boxsize[0] - pos[:, 0]
        if flip[1]:
            pos[:, 1] = boxsize[1] - pos[:, 1]

        # pos -> angle
        pos /= self.comoving_distance

        # self.comm.Barrier()
        if verbose and self.comm.rank == 0:
            print(
                "Particle preprocessing at redshift %.2f takes" % self.redshift,
                time.time() - t,
                "s",
                flush=True,
            )

        t = time.time()

        # paint overdensity
        pm = ParticleMesh(
            Nmesh=[Nmesh, Nmesh],
            BoxSize=[
                boxsize[0] / self.comoving_distance,
                boxsize[1] / self.comoving_distance,
            ],
            resampler=self.resampler,
        )
        layout = pm.decompose(pos)
        delta = pm.paint(pos, layout=layout, resampler=self.resampler)
        del pos
        delta *= (
            pm.Nmesh.prod() * part_boxsize**3 / np.prod(boxsize) / Ntotal
        )  # 1 + delta
        delta -= 1

        # DensityPlane in lenstools: should be 1.5 * H0^2 / c^2 * Omega_m * comoving_dis * delta_covoming_dis / scale_fac * delta
        c = 299792.458
        H0 = 100  ## TODO: Why is this hardcoded?
        self.fac = (
            1.5
            * H0**2
            / c**2
            * self.Omega_m
            * self.comoving_distance
            * boxsize[2]
            * (1 + self.redshift)
        )
        delta *= self.fac

        # self.comm.Barrier()
        if verbose and self.comm.rank == 0:
            print(
                "Particle painting at redshift %.2f takes" % self.redshift,
                time.time() - t,
                "s",
                flush=True,
            )
        t = time.time()

        deltak = delta.r2c(out=Ellipsis)
        # compensate window function
        if self.resampler == "nearest":
            p = 1
        elif self.resampler == "cic":
            p = 2
        elif self.resampler == "tsc":
            p = 3
        deltak = deltak.apply(CompensateWindow(p), out=Ellipsis)
        # lensing potential
        potentialk = deltak.apply(laplace, out=Ellipsis)
        potentialk *= -2.0
        self.potentialk = potentialk

        # self.comm.Barrier()
        if verbose and self.comm.rank == 0:
            print(
                "Lensing potential calculation at redshift %.2f takes" % self.redshift,
                time.time() - t,
                "s",
                flush=True,
            )

        if return_pos:
            return original_pos

    def calculate_deflectionAngles(self, order=1, verbose=True):
        assert (
            self.potentialk is not None
        ), "Needs to calculate / load lensing potential first!"

        # self.comm.Barrier()
        t = time.time()

        self.deflection_x = self.potentialk.apply(gradient(0, order=order)).c2r(
            out=Ellipsis
        )
        self.deflection_y = self.potentialk.apply(gradient(1, order=order)).c2r(
            out=Ellipsis
        )

        # self.comm.Barrier()
        if verbose and self.comm.rank == 0:
            print(
                "Deflection angle calculation at redshift %.2f takes" % self.redshift,
                time.time() - t,
                "s",
                flush=True,
            )

    def calculate_shearMatrix(self, verbose=True):
        assert (self.potentialk is not None) or (self.deflection_x is not None)

        # self.comm.Barrier()
        t = time.time()

        if self.potentialk is not None:
            self.shear_xx = self.potentialk.apply(hessian([0, 0])).c2r(out=Ellipsis)
            self.shear_xy = self.potentialk.apply(hessian([0, 1])).c2r(out=Ellipsis)
            self.shear_yy = self.potentialk.apply(hessian([1, 1])).c2r(out=Ellipsis)

        elif self.deflection_x is not None:
            deflection_xk = self.deflection_x.r2c()
            self.shear_xx = deflection_xk.apply(gradient(0)).c2r(out=Ellipsis)
            self.shear_xy = deflection_xk.apply(gradient(1), out=Ellipsis).c2r(
                out=Ellipsis
            )
            self.shear_yy = (
                self.deflection_y.r2c()
                .apply(gradient(1), out=Ellipsis)
                .c2r(out=Ellipsis)
            )

        # self.comm.Barrier()
        if verbose and self.comm.rank == 0:
            print(
                "Shear matrix calculation at redshift %.2f takes" % self.redshift,
                time.time() - t,
                "s",
                flush=True,
            )

    def deflectionAngles(self, pos):
        pos = pos + self.margin  # (npos, 2)

        if self.deflection_x is None:
            self.calculate_deflectionAngles()

        if self.periodic:
            pos += self.random_shift * self.deflection_x.pm.BoxSize
            pos[:, 0] %= self.deflection_x.pm.BoxSize[0]
            pos[:, 1] %= self.deflection_x.pm.BoxSize[1]

        if self.transpose:
            pos = np.flip(pos, 1)
        if self.flip[0]:
            pos[:, 0] = self.deflection_x.pm.BoxSize[0] - pos[:, 0]
        if self.flip[1]:
            pos[:, 1] = self.deflection_x.pm.BoxSize[1] - pos[:, 1]

        layout = self.deflection_x.pm.decompose(pos)
        pos1 = layout.exchange(pos)

        angle_x = layout.gather(
            self.deflection_x.readout(pos1, resampler=self.resampler)
        )
        angle_y = layout.gather(
            self.deflection_y.readout(pos1, resampler=self.resampler)
        )
        if self.flip[1]:
            angle_y *= -1
        if self.flip[0]:
            angle_x *= -1
        if self.transpose:
            angle_x, angle_y = angle_y, angle_x
        angle = np.concatenate((angle_x[None], angle_y[None]), 0)  # (2, Npos)

        if self.memory_efficient:
            del self.deflection_x, self.deflection_y
            self.deflection_x = None
            self.deflection_y = None
            gc.collect()

        return angle

    def shearMatrix(self, pos):
        pos = pos + self.margin  # (npos, 2)

        if self.shear_xx is None:
            self.calculate_shearMatrix()

        if self.periodic:
            pos += self.random_shift * self.shear_xx.pm.BoxSize
            pos[:, 0] %= self.shear_xx.pm.BoxSize[0]
            pos[:, 1] %= self.shear_xx.pm.BoxSize[1]

        if self.transpose:
            pos = np.flip(pos, 1)
        if self.flip[0]:
            pos[:, 0] = self.shear_xx.pm.BoxSize[0] - pos[:, 0]
        if self.flip[1]:
            pos[:, 1] = self.shear_xx.pm.BoxSize[1] - pos[:, 1]

        layout = self.shear_xx.pm.decompose(pos)
        pos1 = layout.exchange(pos)

        shear_xx = layout.gather(self.shear_xx.readout(pos1, resampler=self.resampler))
        shear_xy = layout.gather(self.shear_xy.readout(pos1, resampler=self.resampler))
        shear_yy = layout.gather(self.shear_yy.readout(pos1, resampler=self.resampler))
        if self.flip[0] ^ self.flip[1]:
            shear_xy *= -1
        if self.transpose:
            shear_xx, shear_yy = shear_yy, shear_xx
        shear = np.concatenate(
            (shear_xx[None], shear_yy[None], shear_xy[None]), 0
        )  # (3, Npos)

        if self.memory_efficient:
            del self.shear_xx, self.shear_xy, self.shear_yy
            self.shear_xx = None
            self.shear_xy = None
            self.shear_yy = None
            gc.collect()

        return shear

    def randomRoll(self):
        if self.periodic:
            self.random_shift = self.comm.bcast(
                np.random.rand(2) if self.comm.rank == 0 else None, root=0
            )

        self.flip = self.comm.bcast(
            np.random.randn(2) > 0 if self.comm.rank == 0 else None, root=0
        )
        self.transpose = self.comm.bcast(
            np.random.randn() > 0 if self.comm.rank == 0 else None, root=0
        )

    def save(self, filename, files=["potentialk"], verbose=True):
        # self.comm.Barrier()
        t = time.time()
        assert set(files).issubset(
            {"potentialk", "deflection", "shear"}
        ), "Can only save potentialk, deflection and shear"

        # save parameters
        if self.comm.rank == 0:
            if not os.path.exists(filename):
                os.mkdir(filename)
            paramdict = {
                "redshift": self.redshift,
                "Omega_m": self.Omega_m,
                "comoving_distance": self.comoving_distance,
                "fac": self.fac,
                "periodic": self.periodic,
                "margin": self.margin,
                "resampler": self.resampler,
                "random_shift": self.random_shift,
                "flip": self.flip,
                "transpose": self.transpose,
            }
            with open(filename + "/param.pkl", "wb") as handle:
                pickle.dump(paramdict, handle)

        if "potentialk" in files:
            assert self.potentialk is not None
            FieldMesh(self.potentialk).save(filename + "/potentialk", mode="complex")

        if "deflection" in files:
            assert (self.deflection_x is not None) and (self.deflection_y is not None)
            FieldMesh(self.deflection_x).save(filename + "/deflection_x")
            FieldMesh(self.deflection_y).save(filename + "/deflection_y")

        if "shear" in files:
            assert (
                (self.shear_xx is not None)
                and (self.shear_xy is not None)
                and (self.shear_yy is not None)
            )
            FieldMesh(self.shear_xx).save(filename + "/shear_xx")
            FieldMesh(self.shear_xy).save(filename + "/shear_xy")
            FieldMesh(self.shear_yy).save(filename + "/shear_yy")

        # self.comm.Barrier()
        if verbose and self.comm.rank == 0:
            print(
                "Savinging lensing plane at redshift %.2f to disk takes"
                % self.redshift,
                time.time() - t,
                "s",
                flush=True,
            )

    def load(self, filename, files=["potentialk"], verbose=True):
        # self.comm.Barrier()
        t = time.time()
        assert set(files).issubset(
            {"potentialk", "deflection", "shear"}
        ), "Can only load potentialk, deflection and shear"
        assert os.path.exists(filename)

        with open(filename + "/param.pkl", "rb") as handle:
            paramdict = pickle.load(handle)

            self.redshift = paramdict["redshift"]
            self.Omega_m = paramdict["Omega_m"]
            self.comoving_distance = paramdict["comoving_distance"]
            self.fac = paramdict["fac"]
            self.periodic = paramdict["periodic"]
            self.margin = paramdict["margin"]
            self.resampler = paramdict["resampler"]
            self.random_shift = paramdict["random_shift"]
            self.flip = paramdict["flip"]
            self.transpose = paramdict["transpose"]

        if "potentialk" in files:
            self.potentialk = BigFileMesh(
                filename + "/potentialk", dataset="Field", comm=self.comm
            ).to_complex_field()

        if "deflection" in files:
            self.deflection_x = BigFileMesh(
                filename + "/deflection_x", dataset="Field", comm=self.comm
            ).to_real_field()
            self.deflection_y = BigFileMesh(
                filename + "/deflection_y", dataset="Field", comm=self.comm
            ).to_real_field()

        if "shear" in files:
            self.shear_xx = BigFileMesh(
                filename + "/shear_xx", dataset="Field", comm=self.comm
            ).to_real_field()
            self.shear_xy = BigFileMesh(
                filename + "/shear_xy", dataset="Field", comm=self.comm
            ).to_real_field()
            self.shear_yy = BigFileMesh(
                filename + "/shear_yy", dataset="Field", comm=self.comm
            ).to_real_field()

        # self.comm.Barrier()
        if verbose and self.comm.rank == 0:
            print(
                "Loading lensing plane at redshift %.2f takes" % self.redshift,
                time.time() - t,
                "s",
                flush=True,
            )


class EmptyLensingPlane:
    def __init__(self, redshift, Omega_m, comoving_distance, fac=None):
        self.redshift = redshift
        self.Omega_m = Omega_m
        self.comoving_distance = comoving_distance
        self.fac = fac

    def deflectionAngles(self, pos):
        return np.zeros_like(pos.T)

    def shearMatrix(self, pos):
        return np.zeros((3, len(pos)))

    def randomRoll(self):
        pass


#######################################################
###############RayTracer class#########################
#######################################################


class RayTracer:

    """
    Class handler of ray tracing operations: it mainly computes the path corrections of light rays that travel through a set of gravitational lenses

    """

    def __init__(self, comm=MPI.COMM_WORLD):
        self.Nlenses = 0
        self.lens = list()
        self.distance = list()
        self.redshift = list()
        self.comm = comm

    def addLens(self, lens):
        """
        Adds a gravitational lens to the ray tracer, either by putting in a lens plane, or by specifying the name of a file which contains the lens specifications

        """

        self.lens.append(lens)
        self.distance.append(lens.comoving_distance)
        self.redshift.append(lens.redshift)
        self.Nlenses += 1

    def randomRoll(self, seed=None):
        """
        Randomly rolls all the lenses in the system along both axes

        :param seed: random seed with which to initialize the generator
        :type seed: int.

        """

        if seed is not None:
            np.random.seed(seed)

        for lens in self.lens:
            lens.randomRoll()

    def reorderLenses(self):
        """
        Reorders the lenses in the ray tracer according to their comoving distance from the observer

        """

        self.lens = [lens for (redshift, lens) in sorted(zip(self.redshift, self.lens))]
        self.redshift.sort()
        self.distance.sort()

    ##################################################################################################################################
    #####################This method solves the nonlinear lensing ODE#################################################################
    #############################(backward ray tracing)###############################################################################
    ##################################################################################################################################

    def shoot(
        self,
        initial_positions,
        z,
        initial_deflection=None,
        weight_planes=None,
        save_intermediate=False,
        IA=0,
        pz=None,
        Fz=None,
        save_Hessian=None,
        save_final_pos=None,
        save_Nplane=None,
        verbose=True,
    ):
        """
        Shots a bucket of light rays from the observer to the sources at redshift z (backward ray tracing), through the system of gravitational lenses, and computes the deflection statistics

        :param initial_positions: initial angular positions of the light ray bucket, according to the observer; if unitless, the positions are assumed to be in radians. initial_positions[0] is x, initial_positions[1] is y
        :type initial_positions: numpy array or quantity

        :param z: redshift of the sources; if an array is passed, a redshift must be specified for each ray, i.e. z.shape==initial_positions.shape[1:]
        :type z: float. or array

        :param initial_deflection: if not None, this is the initial deflection light rays undergo with respect to the line of sight (equivalent to specifying the first derivative IC on the lensing ODE); must have the same shape as initial_positions
        :type initial_deflection: numpy array or quantity

        :param save_intermediate: save the intermediate positions of the rays too
        :type save_intermediate: bool.

        IA=0: no IA
        IA=1: averaged IA signal over nz
        IA=2: IA signal at the position r5xof galaxy

        F(z): function. F(z) = -A_IA * C1 * rho_cr * Omega_m / D(z) * ((1+z)/(1+z0))^eta * (L/L0)^beta
              default: A_IA=1, C1*rho_cr=0.0134, z0=0.3, eta=0, beta=0
              Eq 19 of https://arxiv.org/pdf/1008.3491.pdf
              Eq 13, 14 of https://arxiv.org/pdf/astro-ph/0406275.pdf
        """
        # Sanity check
        assert (
            initial_positions.ndim >= 2 and initial_positions.shape[0] == 2
        ), "initial positions shape must be (2,...)!"
        shape = initial_positions.shape[1:]
        initial_positions = initial_positions.reshape(2, -1)
        assert IA in [0, 1, 2]

        # Allocate arrays for the intermediate light ray positions and deflections

        if initial_deflection is None:
            current_positions = initial_positions.copy()
            current_deflection = np.zeros(initial_positions.shape)
        else:
            assert initial_deflection.shape == initial_positions.shape
            current_deflection = initial_deflection.copy()
            current_positions = initial_positions + initial_deflection

        if weight_planes is not None:
            if weight_planes.ndim == 1:
                weight_planes = weight_planes[None, :]
            weighted_jacobians = np.zeros(
                (len(weight_planes), 4, initial_positions.shape[1])
            )

        # Initial condition for the jacobian is the identity
        current_jacobian = np.outer(
            np.array([1.0, 0.0, 0.0, 1.0]), np.ones(initial_positions.shape[1:])
        ).reshape((4,) + initial_positions.shape[1:])
        current_jacobian_deflection = np.zeros(current_jacobian.shape)

        # Useful to compute the product of the jacobian (2x2 matrix) with the shear matrix (2x2 symmetric matrix)
        dotter = np.zeros((4, 3, 4))
        dotter[
            (0, 0, 1, 1, 2, 2, 3, 3), (0, 2, 0, 2, 2, 1, 2, 1), (0, 2, 1, 3, 0, 2, 1, 3)
        ] = 1

        if IA > 0:
            assert callable(
                Fz
            ), "F(z) function must be provided if IA signal is required"
            if IA == 1:
                IA_shear_tensors = np.zeros((4, initial_positions.shape[1]))
            else:
                IA_shear_tensors = np.zeros((3, initial_positions.shape[1]))

            # normalize and order nz
            if IA == 1:
                assert pz is not None
                pz[:, 1] /= np.sum(pz[:, 1])
                order = np.argsort(pz[:, 0])
                pz = pz[order]

        mask = np.ones(initial_positions.shape[1], dtype=bool)

        # Decide which is the last lens the light rays should cross
        if type(z) == np.ndarray:
            # Check that shapes correspond
            assert z.shape == shape
            z = z.flatten()

            # Check that redshift is not too high given the current lenses
            assert (
                z.max() <= 2 * self.redshift[-1] - self.redshift[-2]
            ), "Given the current lenses you can trace up to redshift {0:.2f}!".format(
                2 * self.redshift[-1] - self.redshift[-2]
            )

            # Compute the number of lenses that each ray should cross
            last_lens_ray = np.sum(z[None] > np.array(self.redshift)[:, None], 0) - 1
            last_lens = self.comm.allreduce(last_lens_ray.max(), op=MPI.MAX)

            if IA != 1:
                mask = last_lens_ray >= 0
                current_deflection = current_deflection[:, mask]
                current_jacobian_deflection = current_jacobian_deflection[:, mask]

        else:
            # Check that redshift is not too high given the current lenses
            assert (
                z < 2 * self.redshift[-1] - self.redshift[-2]
            ), "Given the current lenses you can trace up to redshift {0:.2f}!".format(
                2 * self.redshift[-1] - self.redshift[-2]
            )
            last_lens = np.sum(z > np.array(self.redshift)) - 1

            if save_Hessian and self.comm.rank == 0:
                save_Hessian = open(save_Hessian, "wb")
                if save_Nplane is None:
                    save_Nplane = last_lens

        if save_intermediate:
            assert type(z) != np.ndarray
            all_jacobians = np.zeros((last_lens + 1, 4, initial_positions.shape[1]))
            if IA == 2:
                all_IA_shear_tensors = np.zeros(
                    (last_lens + 1, 3, initial_positions.shape[1])
                )

        # The light rays positions at the k+1 th step are computed according to Xk+1 = Xk + Dk, where Dk is the deflection
        # To stabilize the solution numerically we compute the deflections as Dk+1 = (Ak-1)Dk + Ck*pk where pk is the deflection due to the potential gradient

        # Ordered references to the lenses
        distance = np.array([0.0] + self.distance)
        redshift = np.array([0.0] + self.redshift)

        if last_lens == len(self.distance) - 1:
            distance = np.append(distance, 2 * distance[-1] - distance[-2])
            redshift = np.append(redshift, 2 * redshift[-1] - redshift[-2])

        # This is the main loop that goes through all the lenses
        for k in range(last_lens + 1):
            # Load in the lens
            current_lens = self.lens[k]
            np.testing.assert_approx_equal(
                current_lens.redshift,
                self.redshift[k],
                significant=4,
                err_msg="Loaded lens ({0}) redshift does not match info file specifications {1} neq {2}!".format(
                    k, current_lens.redshift, self.redshift[k]
                ),
            )

            # Log
            if verbose and self.comm.rank == 0:
                print(
                    "Crossing lens {0} at redshift z={1:.3f}".format(
                        k, current_lens.redshift
                    ),
                    flush=True,
                )
            # self.comm.Barrier()
            start = time.time()
            last_timestamp = start

            # Compute the deflection angles and log timestamp
            deflections = current_lens.deflectionAngles(current_positions.T[mask])

            # self.comm.Barrier()
            now = time.time()
            if verbose and self.comm.rank == 0:
                print(
                    "Retrieval of deflection angles completed in {0:.3f}s".format(
                        now - last_timestamp
                    ),
                    flush=True,
                )
            last_timestamp = now

            # Compute the shear matrices and log timestamp
            shear_tensors = current_lens.shearMatrix(current_positions.T[mask])

            # self.comm.Barrier()
            now = time.time()
            if verbose and self.comm.rank == 0:
                print(
                    "Shear matrices retrieved in {0:.3f}s".format(now - last_timestamp),
                    flush=True,
                )
            last_timestamp = now

            #####################################################################################

            # Compute geometrical weight factors
            Ak = (distance[k + 1] / distance[k + 2]) * (
                1.0
                + (distance[k + 2] - distance[k + 1]) / (distance[k + 1] - distance[k])
            )
            Ck = -1.0 * (distance[k + 2] - distance[k + 1]) / distance[k + 2]

            # Compute the position on the next lens and log timestamp
            current_deflection *= Ak - 1
            # self.comm.Barrier()
            now = time.time()
            if verbose and self.comm.rank == 0:
                print(
                    "Geometrical weight factors calculations and deflection scaling completed in {0:.3f}s".format(
                        now - last_timestamp
                    ),
                    flush=True,
                )
            last_timestamp = now

            # Add deflections and log timestamp
            current_deflection += Ck * deflections
            # self.comm.Barrier()
            now = time.time()
            if verbose and self.comm.rank == 0:
                print(
                    "Deflection angles computed in {0:.3f}s".format(
                        now - last_timestamp
                    ),
                    flush=True,
                )
            last_timestamp = now

            # If we are tracing jacobians we need to compute the matrix product with the shear matrix
            current_jacobian_deflection *= Ak - 1

            # This is the part in which the products with the shear matrix are computed
            current_jacobian_deflection += Ck * (
                np.tensordot(dotter, current_jacobian[:, mask], axes=([2], [0]))
                * shear_tensors
            ).sum(1)

            # self.comm.Barrier()
            now = time.time()
            if verbose and self.comm.rank == 0:
                print(
                    "Shear matrix products computed in {0:.3f}s".format(
                        now - last_timestamp
                    ),
                    flush=True,
                )
            last_timestamp = now

            ###########################################################################################

            if type(z) == np.ndarray:
                select = k < last_lens_ray
                last = k == last_lens_ray

                if IA == 1:
                    current_positions += current_deflection
                else:
                    current_positions[:, select] += current_deflection[:, select[mask]]
                    current_positions[:, last] += (
                        current_deflection[:, last[mask]]
                        * (z[None, last] - redshift[k + 1])
                        / (redshift[k + 2] - redshift[k + 1])
                    )

                # We need to add the distortions to the jacobians too
                current_jacobian[:, select] += current_jacobian_deflection[
                    :, select[mask]
                ]
                current_jacobian[:, last] += (
                    current_jacobian_deflection[:, last[mask]]
                    * (z[None, last] - redshift[k + 1])
                    / (redshift[k + 2] - redshift[k + 1])
                )

                # IA
                if IA == 1:
                    select_pz = (pz[:, 0] > redshift[k + 1]) & (
                        pz[:, 0] < redshift[k + 2]
                    )
                    Sij = shear_tensors / current_lens.fac / 2.0
                    delta = Sij[0] + Sij[1]
                    Sij[:2] -= delta / 3.0
                    IA_shear_tensors[0] += (
                        Fz(current_lens.redshift)
                        * (Sij[0] - Sij[1])
                        * np.sum(pz[select_pz, 1])
                    )
                    IA_shear_tensors[1] += (
                        Fz(current_lens.redshift)
                        * 2
                        * Sij[2]
                        * np.sum(pz[select_pz, 1])
                    )
                    # delta * sij
                    IA_shear_tensors[2] += (
                        Fz(current_lens.redshift)
                        * delta
                        * (Sij[0] - Sij[1])
                        * np.sum(pz[select_pz, 1])
                    )
                    IA_shear_tensors[3] += (
                        Fz(current_lens.redshift)
                        * 2
                        * delta
                        * Sij[2]
                        * np.sum(pz[select_pz, 1])
                    )
                elif IA == 2:
                    Sij = shear_tensors[:, last[mask]] / current_lens.fac / 2.0
                    delta = Sij[0] + Sij[1]
                    Sij[:2] -= delta / 3.0
                    IA_shear_tensors[0, last] = Fz(current_lens.redshift) * (
                        Sij[0] - Sij[1]
                    )
                    IA_shear_tensors[1, last] = Fz(current_lens.redshift) * 2 * Sij[2]
                    IA_shear_tensors[2, last] = delta

                # update mask
                if IA != 1:
                    select = select[mask]
                    current_deflection = current_deflection[:, select]
                    current_jacobian_deflection = current_jacobian_deflection[:, select]
                    mask[mask] = select

                    if k == last_lens:
                        assert not mask.any()

            else:
                if k < last_lens:
                    current_positions += current_deflection
                else:
                    current_positions += (
                        current_deflection
                        * (z - redshift[k + 1])
                        / (redshift[k + 2] - redshift[k + 1])
                    )

                if save_Nplane and k == save_Nplane - 1 and save_final_pos:
                    positions = GatherArray(current_positions.T, self.comm, root=0)
                    if self.comm.rank == 0:
                        save_final_pos = open(save_final_pos, "wb")
                        positions.T.astype(np.float16).tofile(
                            save_final_pos
                        )  # (2, Npos)
                        del positions
                    deflection = GatherArray(current_deflection.T, self.comm, root=0)
                    if self.comm.rank == 0:
                        deflection.T.astype(np.float16).tofile(
                            save_final_pos
                        )  # (2, Npos)
                        del deflection
                        save_final_pos.close()

                # We need to add the distortions to the jacobians too
                if k < last_lens:
                    current_jacobian += current_jacobian_deflection
                else:
                    current_jacobian += (
                        current_jacobian_deflection
                        * (z - redshift[k + 1])
                        / (redshift[k + 2] - redshift[k + 1])
                    )

                if IA == 1:
                    select_pz = (pz[:, 0] > redshift[k + 1]) & (
                        pz[:, 0] < redshift[k + 2]
                    )
                    # sij
                    Sij = shear_tensors / current_lens.fac / 2.0
                    delta = Sij[0] + Sij[1]
                    Sij[:2] -= delta / 3.0
                    IA_shear_tensors[0] += (
                        Fz(current_lens.redshift)
                        * (Sij[0] - Sij[1])
                        * np.sum(pz[select_pz, 1])
                    )
                    IA_shear_tensors[1] += (
                        Fz(current_lens.redshift)
                        * 2
                        * Sij[2]
                        * np.sum(pz[select_pz, 1])
                    )
                    # delta * sij
                    IA_shear_tensors[2] += (
                        Fz(current_lens.redshift)
                        * delta
                        * (Sij[0] - Sij[1])
                        * np.sum(pz[select_pz, 1])
                    )
                    IA_shear_tensors[3] += (
                        Fz(current_lens.redshift)
                        * 2
                        * delta
                        * Sij[2]
                        * np.sum(pz[select_pz, 1])
                    )

                elif IA == 2 and k == last_lens:
                    Sij = shear_tensors / current_lens.fac / 2.0
                    delta = Sij[0] + Sij[1]
                    Sij[:2] -= delta / 3.0
                    IA_shear_tensors[0] = Fz(current_lens.redshift) * (Sij[0] - Sij[1])
                    IA_shear_tensors[1] = Fz(current_lens.redshift) * 2 * Sij[2]
                    IA_shear_tensors[2] = delta

                if save_Hessian and save_Nplane and k < save_Nplane:
                    shear_tensors = GatherArray(shear_tensors.T, self.comm, root=0)
                    if self.comm.rank == 0:
                        (100.0 * shear_tensors).T.astype(np.float16).tofile(
                            save_Hessian
                        )  # (3, Npos)
                        del shear_tensors
                        if k == save_Nplane - 1:
                            save_Hessian.close()

            # self.comm.Barrier()
            now = time.time()
            if verbose and self.comm.rank == 0:
                print(
                    "Addition of deflections completed in {0:.3f}s".format(
                        now - last_timestamp
                    ),
                    flush=True,
                )
            last_timestamp = now

            if weight_planes is not None:
                weighted_jacobians += (
                    weight_planes[:, k].reshape(-1, 1, 1) * current_jacobian[None]
                )

            # Save the intermediate positions if option was specified
            if save_intermediate:
                all_jacobians[k] = current_jacobian.copy()
                if IA == 2:
                    Sij = shear_tensors / current_lens.fac / 2.0
                    delta = Sij[0] + Sij[1]
                    Sij[:2] -= delta / 3.0
                    all_IA_shear_tensors[k, 0] = Fz(current_lens.redshift) * (
                        Sij[0] - Sij[1]
                    )
                    all_IA_shear_tensors[k, 1] = Fz(current_lens.redshift) * 2 * Sij[2]
                    all_IA_shear_tensors[k, 2] = delta

            # Log timestamp to cross lens
            # self.comm.Barrier()
            now = time.time()
            if verbose and self.comm.rank == 0:
                print(
                    "Lens {0} at z={1:.3f} crossed in {2:.3f}s".format(
                        k, current_lens.redshift, now - start
                    ),
                    flush=True,
                )

        current_positions = current_positions.reshape(2, *shape)

        # Return the final positions of the light rays (or jacobians)
        convergence = 1.0 - 0.5 * (current_jacobian[0] + current_jacobian[3]).reshape(
            *shape
        )
        shear = np.array(
            [
                0.5 * (current_jacobian[3] - current_jacobian[0]),
                -0.5 * (current_jacobian[1] + current_jacobian[2]),
            ]
        ).reshape(2, *shape)
        if IA > 0:
            IA_shear_tensors = IA_shear_tensors.reshape(-1, *shape)

        if save_intermediate:
            all_convergence = 1.0 - 0.5 * (
                all_jacobians[:, 0] + all_jacobians[:, 3]
            ).reshape(-1, *shape)
            all_shear = np.concatenate(
                (
                    0.5 * (all_jacobians[:, [3]] - all_jacobians[:, [0]]),
                    -0.5 * (all_jacobians[:, [1]] + all_jacobians[:, [2]]),
                ),
                1,
            ).reshape(-1, 2, *shape)
            if IA == 2:
                all_IA_shear_tensors = all_IA_shear_tensors.reshape(
                    last_lens + 1, 3, *shape
                )
                return (
                    all_convergence,
                    all_shear,
                    convergence,
                    shear,
                    all_IA_shear_tensors,
                    IA_shear_tensors,
                    current_positions,
                )
            elif IA == 1:
                return (
                    all_convergence,
                    all_shear,
                    convergence,
                    shear,
                    IA_shear_tensors,
                    current_positions,
                )
            else:
                return all_convergence, all_shear, convergence, shear, current_positions

        elif weight_planes is not None:
            weighted_convergence = 1.0 - 0.5 * (
                weighted_jacobians[:, 0] + weighted_jacobians[:, 3]
            ).reshape(-1, *shape)
            weighted_shear = np.concatenate(
                (
                    0.5 * (weighted_jacobians[:, [3]] - weighted_jacobians[:, [0]]),
                    -0.5 * (weighted_jacobians[:, [1]] + weighted_jacobians[:, [2]]),
                ),
                1,
            ).reshape(-1, 2, *shape)
            if IA == 0:
                return (
                    weighted_convergence,
                    weighted_shear,
                    convergence,
                    shear,
                    current_positions,
                )
            else:
                return (
                    weighted_convergence,
                    weighted_shear,
                    convergence,
                    shear,
                    IA_shear_tensors,
                    current_positions,
                )

        else:
            if IA == 0:
                return convergence, shear, current_positions
            else:
                return convergence, shear, IA_shear_tensors, current_positions


def shear_catalog(
    galaxy_positions,
    z,
    shearMatrix,
    Nmesh,
    FoV,
    plane_redshifts,
    plane_distance,
    Omega_m,
    IA=0,
    pz=None,
    Fz=None,
    resampler="SYM20",
):
    # does not support parallization

    assert IA in [0, 1, 2]

    pm = ParticleMesh(Nmesh=[Nmesh, Nmesh], BoxSize=[FoV, FoV], resampler=resampler)
    field = pm.create(type="real")

    # Initial condition for the jacobian is the identity
    current_jacobian = np.outer(np.array([1.0, 0.0, 0.0, 1.0]), np.ones(Nmesh**2))
    current_jacobian_deflection = np.zeros(current_jacobian.shape)

    convergence = np.zeros(len(galaxy_positions))
    shear = np.zeros((2, len(galaxy_positions)))

    if IA > 0:
        assert callable(Fz), "F(z) function must be provided if IA signal is required"
        if IA == 1:
            IA_shear_tensors = np.zeros((4, Nmesh * Nmesh))
        else:
            IA_shear_tensors = np.zeros((3, len(galaxy_positions)))

        # normalize and order nz
        if IA == 1:
            assert pz is not None
            pz[:, 1] /= np.sum(pz[:, 1])
            order = np.argsort(pz[:, 0])
            pz = pz[order]

    # Decide which is the last lens the light rays should cross

    # Check that redshift is not too high given the current lenses
    assert (
        z.max() <= 2 * plane_redshifts[-1] - plane_redshifts[-2]
    ), "Given the current lenses you can trace up to redshift {0:.2f}!".format(
        2 * plane_redshifts[-1] - plane_redshifts[-2]
    )

    # Compute the number of lenses that each ray should cross
    last_lens_ray = np.sum(z[None] > np.array(plane_redshifts)[:, None], 0) - 1
    last_lens = last_lens_ray.max()

    # Ordered references to the lenses
    distance = np.insert(plane_distance, 0, 0.0)
    redshift = np.insert(plane_redshifts, 0, 0.0)

    if last_lens == len(plane_distance) - 1:
        distance = np.append(distance, 2 * distance[-1] - distance[-2])
        redshift = np.append(redshift, 2 * redshift[-1] - redshift[-2])

    # This is the main loop that goes through all the lenses
    for k in range(last_lens + 1):
        # Compute geometrical weight factors
        Ak = (distance[k + 1] / distance[k + 2]) * (
            1.0 + (distance[k + 2] - distance[k + 1]) / (distance[k + 1] - distance[k])
        )
        Ck = -1.0 * (distance[k + 2] - distance[k + 1]) / distance[k + 2]

        # If we are tracing jacobians we need to compute the matrix product with the shear matrix
        current_jacobian_deflection *= Ak - 1

        # Compute the shear matrices and log timestamp
        if k < len(shearMatrix):
            shear_tensors = shearMatrix[k].astype(np.float64) / 100.0

            # This is the part in which the products with the shear matrix are computed
            current_jacobian_deflection[0] += Ck * (
                current_jacobian[0] * shear_tensors[0]
                + current_jacobian[2] * shear_tensors[2]
            )
            current_jacobian_deflection[1] += Ck * (
                current_jacobian[1] * shear_tensors[0]
                + current_jacobian[3] * shear_tensors[2]
            )
            current_jacobian_deflection[2] += Ck * (
                current_jacobian[2] * shear_tensors[1]
                + current_jacobian[0] * shear_tensors[2]
            )
            current_jacobian_deflection[3] += Ck * (
                current_jacobian[3] * shear_tensors[1]
                + current_jacobian[1] * shear_tensors[2]
            )

        ###########################################################################################

        # read out shear and convergence
        last = k == last_lens_ray
        if last.any():
            jacobian = np.zeros((3, np.sum(last)))
            weight = (z[last] - redshift[k + 1]) / (redshift[k + 2] - redshift[k + 1])

            field[:] = current_jacobian[0].reshape(Nmesh, Nmesh)
            jacobian[0] = field.readout(galaxy_positions[last])
            field[:] = (current_jacobian[1] + current_jacobian[2]).reshape(
                Nmesh, Nmesh
            ) / 2.0
            jacobian[1] = field.readout(galaxy_positions[last])
            field[:] = current_jacobian[3].reshape(Nmesh, Nmesh)
            jacobian[2] = field.readout(galaxy_positions[last])

            field[:] = current_jacobian_deflection[0].reshape(Nmesh, Nmesh)
            jacobian[0] += weight * field.readout(galaxy_positions[last])
            field[:] = (
                current_jacobian_deflection[1] + current_jacobian_deflection[2]
            ).reshape(Nmesh, Nmesh) / 2.0
            jacobian[1] += weight * field.readout(galaxy_positions[last])
            field[:] = current_jacobian_deflection[3].reshape(Nmesh, Nmesh)
            jacobian[2] += weight * field.readout(galaxy_positions[last])

            convergence[last] = 1.0 - 0.5 * (jacobian[0] + jacobian[2])
            shear[:, last] = np.array((0.5 * (jacobian[2] - jacobian[0]), -jacobian[1]))

        # We need to add the distortions to the jacobians too
        current_jacobian += current_jacobian_deflection

        # IA
        if k < len(shearMatrix) and IA > 0 and last.any():
            c = 299792.458
            H0 = 100
            if k == 0:
                thick = (distance[0] + distance[1]) / 2.0
            elif k == len(distance) - 3:
                thick = distance[-1] - distance[-2]
            else:
                thick = (distance[k + 2] - distance[k]) / 2.0
            fac = (
                1.5
                * H0**2
                / c**2
                * Omega_m
                * distance[k + 1]
                * thick
                * (1 + redshift[k + 1])
            )
            Sij = shear_tensors / fac / 2.0
            delta = Sij[0] + Sij[1]
            Sij[:2] -= delta / 3.0
            if IA == 1:
                select_pz = (pz[:, 0] > redshift[k + 1]) & (pz[:, 0] < redshift[k + 2])
                IA_shear_tensors[0] += (
                    Fz(redshift[k + 1]) * (Sij[0] - Sij[1]) * np.sum(pz[select_pz, 1])
                )
                IA_shear_tensors[1] += (
                    Fz(redshift[k + 1]) * 2 * Sij[2] * np.sum(pz[select_pz, 1])
                )
                # delta * sij
                IA_shear_tensors[2] += (
                    Fz(redshift[k + 1])
                    * delta
                    * (Sij[0] - Sij[1])
                    * np.sum(pz[select_pz, 1])
                )
                IA_shear_tensors[3] += (
                    Fz(redshift[k + 1]) * 2 * delta * Sij[2] * np.sum(pz[select_pz, 1])
                )
            elif IA == 2:
                field[:] = (Sij[0] - Sij[1]).reshape(Nmesh, Nmesh)
                IA_shear_tensors[0, last] = Fz(redshift[k + 1]) * field.readout(
                    galaxy_positions[last]
                )
                field[:] = (2 * Sij[2]).reshape(Nmesh, Nmesh)
                IA_shear_tensors[1, last] = Fz(redshift[k + 1]) * field.readout(
                    galaxy_positions[last]
                )
                field[:] = delta.reshape(Nmesh, Nmesh)
                IA_shear_tensors[2, last] = field.readout(galaxy_positions[last])

        print("cross Lensing plane", k)

    if IA == 0:
        return convergence, shear
    elif IA == 1:
        IA_signal = np.zeros((4, len(galaxy_positions)))
        for i in range(4):
            field[:] = IA_shear_tensors[i].reshape(Nmesh, Nmesh)
            IA_signal[i] = field.readout(galaxy_positions)
        return convergence, shear, IA_signal
    elif IA == 2:
        return convergence, shear, IA_shear_tensors


"""
def IA_convergence(shearMatrix, plane_redshifts, plane_distance, Omega_m, weights, Fz):

    IA_shear_tensors = np.zeros((len(weights), 4, shearMatrix.shape[-1]))
    
    #Ordered references to the lenses
    distance = np.insert(plane_distance, 0, 0.)
    redshift = np.insert(plane_redshifts, 0, 0.)
    distance = np.append(distance, 2*distance[-1]-distance[-2])
    redshift = np.append(redshift, 2*redshift[-1]-redshift[-2])
    
    c = 299792.458
    H0 = 100

    #This is the main loop that goes through all the lenses
    for k in range(len(shearMatrix)):
    
        shear_tensors = shearMatrix[k].astype(np.float64) / 100.
        
        if k == 0:
            thick = (distance[0] + distance[1]) / 2.
        elif k == len(distance)-3:
            thick = distance[-1] - distance[-2]
        else:
            thick = (distance[k+2] - distance[k]) / 2.

        fac = 1.5 * H0 ** 2 / c**2 * Omega_m * distance[k+1] * thick * (1+redshift[k+1]) 
        Sij = shear_tensors / fac / 2.
        delta = Sij[0] + Sij[1]
        Sij[:2] -= delta / 3.
        IA_shear_tensors[:,0] += Fz(redshift[k+1]) * (Sij[0] - Sij[1]) * weights[:,[k]] 
        IA_shear_tensors[:,1] += Fz(redshift[k+1]) * 2 * Sij[2] * weights[:,[k]]
        # delta * sij
        IA_shear_tensors[:,2] += Fz(redshift[k+1]) * delta * (Sij[0] - Sij[1]) * weights[:,[k]]
        IA_shear_tensors[:,3] += Fz(redshift[k+1]) * 2 * delta * Sij[2] * weights[:,[k]]

    Nmesh = 4096
    pixel_size = 16/4096/180*np.pi
    gammak = np.fft.fft2(IA_shear_tensors.reshape(len(weights), 4, Nmesh, Nmesh))
    
    l1 = np.fft.fftfreq(Nmesh, d=pixel_size)[None,:].repeat(Nmesh, 0)
    l2 = np.fft.fftfreq(Nmesh, d=pixel_size)[:,None].repeat(Nmesh, 1)
    l_sq = l1**2 + l2**2
    l_sq[0,0] = 1.
    
    kappak = (l1**2 - l2**2) / l_sq * gammak[:,0] + 2*l1*l2 / l_sq * gammak[:,1]
    convergence_NLA = np.fft.ifft2(kappak).real
    kappak = (l1**2 - l2**2) / l_sq * gammak[:,2] + 2*l1*l2 / l_sq * gammak[:,3]
    convergence_bta = np.fft.ifft2(kappak).real
    
    return convergence_NLA, convergence_bta
"""


def IA_convergence(shearMatrix, plane_redshifts, plane_distance, Omega_m, pz, Fz):
    IA_shear_tensors = np.zeros((len(pz) - 1, 2, shearMatrix.shape[-1]))
    convergence_NLA = np.zeros((len(pz) - 1, shearMatrix.shape[-1]))

    # Ordered references to the lenses
    distance = np.insert(plane_distance, 0, 0.0)
    redshift = np.insert(plane_redshifts, 0, 0.0)

    c = 299792.458
    H0 = 100

    # This is the main loop that goes through all the lenses
    for k in range(len(shearMatrix)):
        shear_tensors = shearMatrix[k].astype(np.float64) / 100.0

        if k == 0:
            thick = (distance[1] + distance[2]) / 2.0
            z1 = 0
            z2 = (redshift[k + 1] + redshift[k + 2]) / 2.0
        else:
            thick = (distance[k + 2] - distance[k]) / 2.0
            z1 = (redshift[k] + redshift[k + 1]) / 2.0
            z2 = (redshift[k + 1] + redshift[k + 2]) / 2.0

        fac = (
            1.5
            * H0**2
            / c**2
            * Omega_m
            * distance[k + 1]
            * thick
            * (1 + redshift[k + 1])
        )
        Sij = shear_tensors / fac / 2.0
        delta = Sij[0] + Sij[1]
        # Sij[:2] -= delta / 3.
        select = (pz[0] >= z1) & (pz[0] < z2)
        weight = Fz(redshift[k + 1]) * np.sum(pz[1:, select], 1)[:, None]
        # weight = Fz(redshift[k+1]) * (np.sum(select) * np.sum(pz[1:,select]**2, 1)[:,None]) ** 0.5
        # select = np.nonzero((pz[0] >= z1) & (pz[0] < z2))[0]
        # weight = np.sum(np.array([Fz(pz[0,i]) * pz[1:,i] for i in select]), 0)[:,None]
        convergence_NLA += weight * delta
        # delta * sij
        IA_shear_tensors[:, 0] += weight * delta * (Sij[0] - Sij[1])
        IA_shear_tensors[:, 1] += (weight * 2) * delta * Sij[2]

        # print('cross Lensing plane', k)

    Nmesh = 4096
    pixel_size = 16 / 4096 / 180 * np.pi
    gammak = np.fft.fft2(IA_shear_tensors.reshape(len(pz) - 1, 2, Nmesh, Nmesh))

    l2 = np.fft.fftfreq(Nmesh, d=pixel_size)[None, :].repeat(Nmesh, 0)
    l1 = np.fft.fftfreq(Nmesh, d=pixel_size)[:, None].repeat(Nmesh, 1)
    l_sq = l1**2 + l2**2
    l_sq[0, 0] = 1.0

    kappak = (l1**2 - l2**2) / l_sq * gammak[:, 0] + 2 * l1 * l2 / l_sq * gammak[
        :, 1
    ]
    convergence_bta = np.fft.ifft2(kappak).real

    return convergence_NLA, convergence_bta


def save_convergence_planes(
    shearMatrix, plane_redshifts, plane_distance, Omega_m, Fz, save_dir
):
    # does not support parallization

    c = 299792.458
    H0 = 100
    Nmesh = 4096
    pixel_size = 16 / 4096 / 180 * np.pi
    dtype = np.float16

    # Initial condition for the jacobian is the identity
    current_jacobian = np.outer(np.array([1.0, 0.0, 0.0, 1.0]), np.ones(Nmesh**2))
    current_jacobian_deflection = np.zeros(current_jacobian.shape)

    # Ordered references to the lenses
    distance = np.insert(plane_distance, 0, 0.0)
    redshift = np.insert(plane_redshifts, 0, 0.0)

    distance = np.append(distance, 2 * distance[-1] - distance[-2])
    redshift = np.append(redshift, 2 * redshift[-1] - redshift[-2])

    save_convergence = open(save_dir + "convergence", "wb")
    save_IAconvergence = open(save_dir + "IAconvergence", "wb")
    save_IAbTAconvergence = open(save_dir + "IAbTAconvergence", "wb")

    # This is the main loop that goes through all the lenses
    for k in range(len(plane_redshifts)):
        # Compute geometrical weight factors
        Ak = (distance[k + 1] / distance[k + 2]) * (
            1.0 + (distance[k + 2] - distance[k + 1]) / (distance[k + 1] - distance[k])
        )
        Ck = -1.0 * (distance[k + 2] - distance[k + 1]) / distance[k + 2]

        # If we are tracing jacobians we need to compute the matrix product with the shear matrix
        current_jacobian_deflection *= Ak - 1

        # Compute the shear matrices and log timestamp
        if k < len(shearMatrix):
            shear_tensors = shearMatrix[k].astype(np.float64) / 100.0

            # This is the part in which the products with the shear matrix are computed
            current_jacobian_deflection[0] += Ck * (
                current_jacobian[0] * shear_tensors[0]
                + current_jacobian[2] * shear_tensors[2]
            )
            current_jacobian_deflection[1] += Ck * (
                current_jacobian[1] * shear_tensors[0]
                + current_jacobian[3] * shear_tensors[2]
            )
            current_jacobian_deflection[2] += Ck * (
                current_jacobian[2] * shear_tensors[1]
                + current_jacobian[0] * shear_tensors[2]
            )
            current_jacobian_deflection[3] += Ck * (
                current_jacobian[3] * shear_tensors[1]
                + current_jacobian[1] * shear_tensors[2]
            )

        ###########################################################################################

        # We need to add the distortions to the jacobians too
        current_jacobian += current_jacobian_deflection

        convergence = 1.0 - 0.5 * (current_jacobian[0] + current_jacobian[3]).reshape(
            Nmesh, Nmesh
        )  # plane_redshifts[k+1]
        convergence = np.mean(convergence.reshape(2048, 2, 2048, 2), (1, 3))
        convergence = np.mean(convergence.reshape(1024, 2, 1024, 2), (1, 3))
        convergence.astype(dtype).tofile(save_convergence)
        del convergence

        # IA
        if k < len(shearMatrix):
            if k == 0:
                thick = (distance[0] + distance[1]) / 2.0
            else:
                thick = (distance[k + 2] - distance[k]) / 2.0
            fac = (
                1.5
                * H0**2
                / c**2
                * Omega_m
                * distance[k + 1]
                * thick
                * (1 + redshift[k + 1])
            )
            Sij = shear_tensors / fac / 2.0
            delta = Sij[0] + Sij[1]

            convergence = Fz(redshift[k + 1]) * delta.reshape(Nmesh, Nmesh)
            convergence = np.mean(convergence.reshape(2048, 2, 2048, 2), (1, 3))
            convergence = np.mean(convergence.reshape(1024, 2, 1024, 2), (1, 3))
            convergence.astype(dtype).tofile(save_IAconvergence)
            del convergence

            # delta * sij
            IA_shear_tensors = np.zeros((2, Nmesh * Nmesh))
            IA_shear_tensors[0] = Fz(redshift[k + 1]) * delta * (Sij[0] - Sij[1])
            IA_shear_tensors[1] = Fz(redshift[k + 1]) * 2 * delta * Sij[2]
            del delta, Sij

            gammak = np.fft.fft2(IA_shear_tensors.reshape(2, Nmesh, Nmesh))
            del IA_shear_tensors

            l2 = np.fft.fftfreq(Nmesh, d=pixel_size)[None, :].repeat(Nmesh, 0)
            l1 = np.fft.fftfreq(Nmesh, d=pixel_size)[:, None].repeat(Nmesh, 1)
            l_sq = l1**2 + l2**2
            l_sq[0, 0] = 1.0

            kappak = (l1**2 - l2**2) / l_sq * gammak[
                0
            ] + 2 * l1 * l2 / l_sq * gammak[1]
            del gammak, l1, l2, l_sq
            convergence = np.fft.ifft2(kappak).real
            del kappak
            convergence = np.mean(convergence.reshape(2048, 2, 2048, 2), (1, 3))
            convergence = np.mean(convergence.reshape(1024, 2, 1024, 2), (1, 3))
            convergence.astype(dtype).tofile(save_IAbTAconvergence)
            del convergence

        # print('cross Lensing plane', k, flush=True)

    save_convergence.close()
    save_IAconvergence.close()
    save_IAbTAconvergence.close()

    return
