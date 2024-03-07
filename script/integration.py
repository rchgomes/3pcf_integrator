import time
import numpy as np
import matplotlib.pyplot as plt
from bihalofit import bihalofit
from funcs import f_h3, f_psi3, f_psi1, f_psi2, transform_gamma
from compute_3pcf import compute_3pcf, compute_3pcf_ro, plot_3pcf
from run_classy import run_classy
from halo_model import halo_model_bispectrum

'''Specify the cosmology'''
Omega_b = 0.05
Omega_m = 0.308
Omega_cdm = Omega_m - Omega_b
h = 0.678
A_s = 2.1e-9
n_s = 0.968

k_max = 50

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

mynz = np.loadtxt("bin_04_desy3_source_nz.dat")

''' equilateral: input in acrmins'''
my_equilateral_xs = np.logspace(np.log10(3),np.log10(35), num=10)
my_u_values = 1*np.ones_like(my_equilateral_xs)
my_v_values = np.zeros_like(my_equilateral_xs)

'''isosceles: input in acrmins'''
my_angles = np.linspace(2,178, num=20)
u_2 = np.ones_like(my_angles)
v_2 = np.zeros_like(my_angles)
new_d2 = 4.0*np.ones_like(u_2)
for i in range(len(my_angles)):
    if my_angles[i] < 60:
        u_2[i] = np.sqrt(2*(1-np.cos(my_angles[i]*np.pi/180)))
    else:
        v_2[i] = np.sqrt(2*(1-np.cos(my_angles[i]*np.pi/180)))-1

#compute_3pcf(params, mynz, my_equilateral_xs, my_u_values, my_v_values, "new_scenario_test_10apr_neval50000", neval=50000)
#compute_3pcf_ro(params, mynz, my_equilateral_xs, my_u_values, my_v_values, "new_scenario_test_10apr_neval50000_ro", neval=50000)

my_file_list = ["Gamma1_real_new_scenario_angle_test_10apr_neval50000.npy"]
my_label_list = ["typical order"]
plot_3pcf(my_file_list, my_label_list, my_angles, "$\Gamma^1$", no_log=True)

compute_3pcf(params, mynz, new_d2, u_2, v_2, "new_scenario_angle_test_10apr_neval10000", neval=10000)