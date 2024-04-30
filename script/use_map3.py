import numpy as np
from compute_3pcf import compute_map3
import time

h = 0.673
"Maximum scale in units of h/Mpc"
k_max = 100

cosmogrid_params = {
    #including As to match sigma8=0.84
             'output':'mPk',
             'non linear':'halofit',
             'Omega_b':0.0493,
             'Omega_cdm':0.2107,
             'h':0.673,
             'A_s':3.0557e-9,
             'n_s':0.9649,
             'P_k_max_1/Mpc':k_max*h,
             'z_max_pk':10.
}

my_k_reduced = np.logspace(-4, np.log10(k_max), num=1000)
z = np.linspace(0,3,100)

#theta_1 = np.logspace(np.log10(0.5), np.log10(90), num = 30)
#theta_2 = np.logspace(np.log10(0.5), np.log10(90), num = 30)
#theta_3 = np.logspace(np.log10(0.5), np.log10(90), num = 30)
theta_1 = [7,14,25,50]
theta_2 = [7,14,25,50]
theta_3 = [7,14,25,50]

dndz1 = np.load("bin_01_desy3_source_nz_FROM_DV_DEC7.npy")
dndz2 = np.load("bin_02_desy3_source_nz_FROM_DV_DEC7.npy")
dndz3 = np.load("bin_03_desy3_source_nz_FROM_DV_DEC7.npy")
dndz4 = np.load("bin_04_desy3_source_nz_FROM_DV_DEC7.npy")

compute_map3(cosmogrid_params, dndz1, theta_1, theta_2, theta_3, "ModelMap3FromBispec_15Mar_zbin111_COSMOGRID", lmax=10000, niter = 7, neval=200000)
compute_map3(cosmogrid_params, dndz2, theta_1, theta_2, theta_3, "ModelMap3FromBispec_15Mar_zbin222_COSMOGRID", lmax=10000, niter = 7, neval=200000)
compute_map3(cosmogrid_params, dndz3, theta_1, theta_2, theta_3, "ModelMap3FromBispec_15Mar_zbin333_COSMOGRID", lmax=10000, niter = 7, neval=200000)
compute_map3(cosmogrid_params, dndz4, theta_1, theta_2, theta_3, "ModelMap3FromBispec_15Mar_zbin444_COSMOGRID", lmax=10000, niter = 7, neval=200000)

#compute_3pcf_ro(params, dndz, d2, u_theory, v_theory, "Theory_d2eq3_neval10e5_isoc", neval=100000)
#compute_3pcf_ro(params, dndz, d2, u_theory, v_theory, "Theory_d2eq3_neval10e5_almost_isoc", neval=100000)

print(safb)
#my_data_set = np.ndarray(shape=(len(sobol),8))
my_data_set = []
new_sample = []
count = 0

for i in range(5000):
    print(sobol[i])
    timee = time.time()
    if sobol[i][3] < 0.89:
        try_run = do_run(sobol[i])
        if try_run[0] != 10e20:
            my_data_set.append(try_run)
            new_sample.append(sobol[i])
            count += 1
    print(i, count, time.time()-timee)

my_data_set = np.array(my_data_set)
new_sobol = np.array(new_sample)

np.save("Training_set_y", my_data_set)
np.save("New_sobol_y", new_sobol)