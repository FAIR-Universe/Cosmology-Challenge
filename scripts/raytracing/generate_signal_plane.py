from nbodykit.cosmology import Cosmology
from background import MatterDominated
from raytracing import save_convergence_planes
import numpy as np
import time
import gc
from mpi4py import MPI
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--simulation_start', type=int, default=0)
    parser.add_argument('--simulation_end', type=int, default=101)
    parser.add_argument('--realization_start', type=int, default=0)
    parser.add_argument('--realization_end', type=int, default=100)
    parser.add_argument('--base_dir', type=str, default='/pscratch/sd/b/biwei/HSC/')
    args = parser.parse_args()
    params = vars(args)

    #Nsim = 1 
    Nsim = params['simulation_end'] - params['simulation_start']
    Nreal = params['realization_end']-params['realization_start']
    # hyperparameters
    all_Omega_m = np.loadtxt('../nbody/cosmology.txt')[params['simulation_start']:params['simulation_end'],0]
    all_sigma_8 = np.loadtxt('../nbody/cosmology.txt')[params['simulation_start']:params['simulation_end'],1]
    #Omega_m = 0.279
    #sigma_8 = 0.82
    #simulation = 'mock'
    base_dir = params['base_dir']
    angle = 16 / 180 * np.pi
    
    # constant
    h = 0.7
    Omega_b = 0.046
    n_s = 0.97
    comm = MPI.COMM_WORLD

    redshift = np.array([0.01674168, 0.0506174 , 0.08504611, 0.12006466, 0.15571175,
                         0.19202803, 0.22905622, 0.26684124, 0.30543036, 0.34487332,
                         0.38522251, 0.42653314, 0.46886342, 0.51227478, 0.55683209,
                         0.60260389, 0.64966266, 0.69808513, 0.74795256, 0.79935109,
                         0.85237212, 0.90711273, 0.96367608, 1.02217196, 1.08271726,
                         1.1454366 , 1.21046296, 1.27793838, 1.34801472, 1.42085458,
                         1.49663214, 1.57553433, 1.65776187, 1.74353062, 1.83307297,
                         1.92663943, 2.02450038, 2.12694802, 2.23429857, 2.34689472,
                         2.46510834, 2.58934359, 2.72004039, 2.85767826, 3.00278084,
                         3.15592076, 3.31772541, 3.48888333, 3.67015156, 3.86236406,
                         4.06644131, 4.2834014 , 4.51437274])

    for index in range(Nsim*Nreal):

        if index % comm.size == comm.rank:

            t = time.time()
            
            simulation = index // Nreal + params['simulation_start']
            realization = index % Nreal + params['realization_start']

            Omega_m = all_Omega_m[simulation-params['simulation_start']]
            sigma_8 = all_sigma_8[simulation-params['simulation_start']]

            # cosmology
            cosmology = Cosmology(h=h, Omega0_b=Omega_b, Omega0_cdm=Omega_m-Omega_b, n_s=n_s)
            Omega_ncdm = cosmology.Omega0_ncdm_tot - cosmology.Omega0_pncdm_tot + cosmology.Omega0_dcdm
            cosmology = Cosmology(h=h, Omega0_b=Omega_b, Omega0_cdm=Omega_m-Omega_ncdm-Omega_b, n_s=n_s)
            
            comoving_dis = np.array([cosmology.comoving_distance(z) for z in redshift])

            background = MatterDominated(Omega0_m=Omega_m)
            Fz = lambda z: - 0.013873073650776856 * Omega_m / background.D1(1./(1+z))
            
            shearMatrix = np.memmap(base_dir + 'shearMatrix/' + str(simulation) + '_realization' + str(realization) + '_Nmesh4096_plane43_shearMatrix_coherent', dtype=np.float16).reshape(43,3,4096*4096)

            save_dir = base_dir + 'kappa_maps/' + str(simulation) + '_realization' + str(realization) + '_Nmesh1024_plane43_'

            save_convergence_planes(shearMatrix, redshift, comoving_dis, Omega_m, Fz, save_dir)

            #print('Finished realization %d' % realization, 'Time:', time.time()-t, flush=True)
            print('Finished cosmology %d realization %d' % (simulation, realization), 'Time:', time.time()-t, flush=True)

            del shearMatrix
            gc.collect()

if __name__ == "__main__":
    main()

