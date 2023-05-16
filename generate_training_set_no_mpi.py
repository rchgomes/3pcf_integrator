import numpy as np
from compute_3pcf import compute_3pcf_ro_gamma0only
import time

lhs_samples = np.load("lhs_samples_128.npy")

#from mpi4py import MPI


def do_some_runs(i):

    marker = time.time()
    index = i
    omega_m = lhs_samples[index][0]
    as_param = lhs_samples[index][1]

    Omega_b = 0.05
    h = 0.678
    n_s = 0.968
    k_max = 50

    params = {
            'output': 'mPk',
            'non linear': 'halofit',
            'Omega_b': Omega_b,
            'Omega_cdm': omega_m-Omega_b,
            'h': h,
            'A_s': as_param,
            'n_s': n_s,
            'P_k_max_1/Mpc': k_max * h,
            'z_max_pk': 10.
    }

    mynz = np.loadtxt("bin_04_desy3_source_nz.dat")

    '''isosceles: input in acrmins'''
    #my_angles = np.linspace(2, 178, num=20)
    #u_2 = np.ones_like(my_angles)
    u_2 = np.linspace(0.0625, 0.9375, num=15)
    #v_2 = np.zeros_like(my_angles)
    v_2 = 0.05 * np.ones_like(u_2)
    #new_d2 = 4.0 * np.ones_like(u_2)
    new_d2 = 15.0 * np.ones_like(u_2)
    #for ii in range(len(my_angles)):
    #    if my_angles[ii] < 60:
    #        u_2[ii] = np.sqrt(2 * (1 - np.cos(my_angles[ii] * np.pi / 180)))
    #    else:
    #        v_2[ii] = np.sqrt(2 * (1 - np.cos(my_angles[ii] * np.pi / 180))) - 1

    compute_3pcf_ro_gamma0only(params, mynz, new_d2, u_2, v_2, "lhs_r15_" + str(index), neval=70000,
                        baryons=False, model='bihalofit')
    print("done", i, time.time()-marker)
# e.g., run the i-th rho stat. here you have to load the right catalog and run treecorr.

#run_count = 0
#number_or_runs = 128  # e.g. 6 like the rho stats you need to compute.
#while run_count < number_or_runs:
#    comm = MPI.COMM_WORLD
#    if run_count + comm.rank < number_or_runs:
#        do_some_runs(run_count + comm.rank)
#    run_count += comm.size
#    comm.bcast(run_count, root=0)
#    comm.Barrier()

for i in range(128):
    do_some_runs(i)