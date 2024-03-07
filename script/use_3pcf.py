import numpy as np
from compute_3pcf import compute_3pcf_ro, compute_3pcf_ro_gamma0only
import time

#d2_t = 150*np.ones(50)
#omega_m_t = 0.279*np.ones(50)
#As_t = 2.1822215e-9*np.ones(50)
#w0_t = -1*np.ones(50)
#h_t = 0.7*np.ones(50)
#z_t = np.ones(50)

'''Specify the cosmology'''
Omega_b = 0.046
Omega_m = 0.279
Omega_cdm = Omega_m - Omega_b
h = 0.7
A_s = 2.1822215e-9 #sigma8=0.82
n_s = 0.97

"Maximum scale in units of h/Mpc"
k_max = 100

params = {
             'output':'mPk',
             'non linear':'halofit',
             'Omega_b':Omega_b,
             'Omega_cdm':Omega_cdm,
             'h':h,
             'A_s':A_s,
             'n_s':n_s,
             'P_k_max_1/Mpc':k_max*h,
             'z_max_pk':10.
}

cosmogrid_params = {
    #except As
             'output':'mPk',
             'non linear':'halofit',
             'Omega_b':0.0493,
             'Omega_cdm':0.2107,
             'h':0.673,
             'A_s':2.1e-9,
             'n_s':0.9649,
             'P_k_max_1/Mpc':k_max*h,
             'z_max_pk':10.
}

my_k_reduced = np.logspace(-4, np.log10(k_max), num=1000)
z = np.linspace(0,3,100)

#u = np.ndarray(shape=(35))
#v = np.ndarray(shape=(35))
#u[:15] = np.linspace(0.03125,0.96875, num=15)
#u[15:] = 1*np.ones(shape=(20))
#v[:15] = 0*np.ones(shape=(15))
##v[15:25] = np.linspace(0.0363636, 0.7636363, num=10)
#v[25:] = np.linspace(0.8090909, 0.99090909, num=10)

#d2 = 5*np.ones(shape=(35))

#x2 = d2 * np.pi / (60 * 180)
#x3 = u * x2
#x1 = v * x3 + x2

#print("x", x1, x2, x3)
#print(d2, u, v)

u = np.ndarray(shape=(35))
v = np.ndarray(shape=(35))
u[:15] = np.linspace(0.03125,0.96875, num=15)
u[15:] = 1*np.ones(shape=(20))
v[:15] = 0*np.ones(shape=(15))
v[15:25] = np.linspace(0.0363636, 0.7636363, num=10)
v[25:] = np.linspace(0.8090909, 0.99090909, num=10)

u_theory = np.ndarray(shape=(50))
v_theory = np.ndarray(shape=(50))
u_theory[:20] = np.linspace(0.0001,0.9999, num=20)
u_theory[20:] = 1*np.ones(shape=(30))
v_theory[:20] = 0*np.ones(shape=(20))
v_theory[20:35] = np.linspace(0.0001, 0.85, num=15)
v_theory[35:] = np.linspace(0.85, 0.99999, num=15)

d2 = 250*np.ones(shape=(50))

x2 = d2 * np.pi / (60 * 180)
x3 = u_theory * x2
x1 = v_theory * x3 + x2

print("x", x1, x2, x3)
dndz = np.loadtxt("bin_04_desy3_source_nz.dat")

dndz1 = np.load("bin_01_desy3_source_nz_FROM_DV_DEC7.npy")
dndz2 = np.load("bin_02_desy3_source_nz_FROM_DV_DEC7.npy")
dndz3 = np.load("bin_03_desy3_source_nz_FROM_DV_DEC7.npy")
dndz4 = np.load("bin_04_desy3_source_nz_FROM_DV_DEC7.npy")

#compute_3pcf_ro_gamma0only(params, dndz, d2, u, v, "tipota")

#print(adfs)
#compute_3pcf_ro(params, dndz, d2, u_theory, v_theory, "Jul17_d2eq250_cosmotypical", neval=100000)

d2dv = np.array([11.00, 13.08, 15.56, 18.50, 22.00, 26.16, 31.11, 37.00, 44.00, 52.33, 62.23, 74.01])
udv = 0.8799*np.ones(shape=(12))
vdv = 0.0502*np.ones(shape=(12))

compute_3pcf_ro(cosmogrid_params, dndz1, d2dv, udv, vdv, "Dec7_test_partial_dv_zbin1", neval=100000)
compute_3pcf_ro(cosmogrid_params, dndz2, d2dv, udv, vdv, "Dec7_test_partial_dv_zbin2", neval=100000)
compute_3pcf_ro(cosmogrid_params, dndz3, d2dv, udv, vdv, "Dec7_test_partial_dv_zbin3", neval=100000)
compute_3pcf_ro(cosmogrid_params, dndz4, d2dv, udv, vdv, "Dec7_test_partial_dv_zbin4", neval=100000)

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