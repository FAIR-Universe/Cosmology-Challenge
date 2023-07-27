""" Copied from: https://github.com/biweidai/PGD/blob/main/PGD.py#L56
"""

import numpy as np
from pmesh.pm import ParticleMesh
from fastpm.force.kernels import gradient, laplace
from nbodykit.cosmology import Planck15


def PGD(cat, alpha, kl, ks, fpm=None, Omega_m0=0.2726):
    """
    improve the small scale matter distribution of quasi N-body simulations with potential gradient descent method

    Parameters
    ----------
    cat    :    "nbodykit" ArrayCatalog object
        particle Position
    alpha    :    float
        the parameter determines the length of the displacements
    kl    :    float
        the large scale cut. Scales larger than kl will be damped.
    ks    :    float
        the small scale smoothing scale
    fpm    :    "pmesh" Particle Mesh object
        used for calculating the displacement

    Returns
    -------
    S    :    array_like
        the displacement at cat['Position']
    """

    if fpm is None:
        B = np.ceil(2. * cat.attrs['BoxSize'] * ks / (2. * np.pi * cat.csize ** (1./3.)))
        mask = B < 1
        B[mask] = 1
        fpm = ParticleMesh(Nmesh=B*cat.attrs['Nmesh'], BoxSize=cat.attrs['BoxSize'], comm=cat.comm, resampler='tsc')

    nbar = 1.0 * cat.csize / fpm.Nmesh.prod()
    H0 = 100.

    X = cat['Position'].compute() 
    X[...] %= cat.attrs['BoxSize']

    layout = fpm.decompose(X)
    X1 = layout.exchange(X)

    rho = fpm.create(mode="real")
    rho.paint(X1, hold=False)
    rho[...] /= nbar # 1 + delta

    delta_k = rho.r2c(out=Ellipsis)
    S = alpha / H0**2 * layout.gather(shortrangePM(X1, delta_k, kl, ks, factor=1.5 * Omega_m0))
        
    return S



def EGD(cat, gamma, beta, r_smth=0.1, BaryonFraction=None, fpm=None):
    """
    effectively model the baryonic effect in N-body simulations with enthalpy gradient descent method

    Parameters
    ----------
    cat    :    "nbodykit" ArrayCatalog object
        particle Position
    gamma    :    float
        the index in the power law equation of state
    beta    :    float
        the parameter determines the length of the displacements
    r_smth    :    float
        the small scale smoothing scale
    BaryonFraction    :    float
        if not provided, use Planck15.Ob0 / Planck15.Om0
    fpm    :    "pmesh" Particle Mesh object
        used for calculating the displacement

    Returns
    -------
    S    :    array_like
        the displacement at cat['Position']
    """

    assert gamma > 1

    if fpm is None:
        B = np.ceil(cat.attrs['BoxSize'] / (r_smth * cat.csize ** (1./3.)))
        mask = B < 1
        B[mask] = 1
        fpm = ParticleMesh(Nmesh=B*cat.attrs['Nmesh'], BoxSize=cat.attrs['BoxSize'], comm=cat.comm)

    if BaryonFraction is None:
        BaryonFraction = Planck15.Ob0 / Planck15.Om0

    nbar = 1.0 * cat.csize / fpm.Nmesh.prod()
    H0 = 100.

    X = cat['Position'].compute()
    X[...] %= cat.attrs['BoxSize']

    layout = fpm.decompose(X)
    X1 = layout.exchange(X)

    rho = fpm.create(mode="real")
    rho.paint(X1, hold=False)
    rho[...] /= nbar # 1 + delta

    rho = rho.r2c(out=Ellipsis).apply(Gaussian(r_smth), out=Ellipsis).c2r(out=Ellipsis)

    select = np.array(rho) < 0
    rho[select] = 0

    Sb = layout.gather(- beta / H0**2 * HPM(rho, gamma, X1))
            
    ran = np.random.random(size=len(X))
    DM = ran > BaryonFraction
    Sb[DM] = 0

    return Sb


def Gaussian(r_smth):
    def kernel(k, v):
        kk = sum(ki ** 2 for ki in k)
        return v * np.exp(-kk * r_smth**2 / 2.)
    return kernel

def shortrange(kl, ks):
    def kernel(k, v):
        kk = sum(ki ** 2 for ki in k)
        kl2 = kl**2
        ks4 = ks**4
        mask = (kk == 0).nonzero()
        kk[mask] = 1
        b = v * np.exp(-kl2 / kk) * np.exp(- kk**2 / ks4)
        b[mask] = 0
        return b
    return kernel

def shortrangePM(x, delta_k, kl, ks, factor):

    f = np.empty_like(x)

    pot_k = delta_k.apply(laplace, out=Ellipsis) \
                  .apply(shortrange(kl, ks), out=Ellipsis)

    for d in range(x.shape[1]):
        force_d = pot_k.apply(gradient(d)) \
                  .c2r(out=Ellipsis)
        force_d.readout(x, out=f[..., d])

    f[...] *= factor

    return f

def HPM(rho, gamma, x, T0=1e4, muH=1.67372e-27):
    """
    Calculate the hydrodynamic force following HPM method.

    Parameters
    ----------
    rho    :   "pmesh" realfield object
        1 + delta
    gamma    :    float
        the index in the power law equation of state
    x    :    array_like
        particle positions to calculate the force
    T0    :    float
        characteristic temperature in the equation of state
    muH    :    float
        the averaged atomic mass of gas

    Returns
    -------
    f    :    array_like
        the hydrodynamic force at x. 
    """
    KB, H0 = 1.38064852e-29, 100.

    rho[...] **= (gamma - 1.)
    rho[...] *= KB * T0 * gamma / muH / (gamma-1)
    enthalpy_k = rho.r2c(out=Ellipsis)

    f = np.empty_like(x)

    for d in range(x.shape[1]):
        force_d = enthalpy_k.apply(gradient(d)) \
                  .c2r(out=Ellipsis)
        force_d.readout(x, out=f[..., d])

    return f
