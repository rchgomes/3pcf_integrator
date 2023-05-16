import numpy as np
from compute_3pcf import compute_3pcf_for_emulator
import time

sobol = np.load("sobol_samples_4.npy")

print(sobol[:,0])
print(sobol[50])
print(qfej)
#from mpi4py import MPI
#print(sobol)
def do_run(sobol_sample):

    omega_m = sobol_sample[0]
    as_param = sobol_sample[1]
    w0 = sobol_sample[2]
    Omega_b = 0.05
    h = sobol_sample[3]
    n_s = 0.968
    k_max = 50

    d2 = sobol_sample[7]
    u = sobol_sample[5]
    v = sobol_sample[6]
    z = sobol_sample[4]

    params = {
            'output': 'mPk',
            'non linear': 'halofit',
            'Omega_b': Omega_b,
            'Omega_cdm': omega_m-Omega_b,
            'Omega_Lambda': 1-omega_m,
            'h': h,
            'w0_fld': w0,
            'A_s': as_param,
            'n_s': n_s,
            'P_k_max_1/Mpc': k_max * h,
            'z_max_pk': 10.
    }

    output = compute_3pcf_for_emulator(params, d2, u, v, z, neval=70000, baryons=False, model='bihalofit')
    return(output)

my_data_set = np.ndarray(shape=(len(sobol),8))

for i in range(len(sobol)):
    print(sobol[i])
    timee = time.time()
    my_data_set[i] = do_run(sobol[i])
    print(i, time.time()-timee)

np.save("Training_set_y", my_data_set)