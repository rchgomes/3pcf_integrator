import time
import numpy as np
import matplotlib.pyplot as plt
from bihalofit import bihalofit
from funcs import f_h3, f_psi3, f_psi1, f_psi2, transform_gamma
from run_classy import run_classy
from halo_model import halo_model_bispectrum

'''Specify the cosmology'''
Omega_b = 0.05
Omega_m = 0.308
Omega_cdm = Omega_m - Omega_b
h = 0.678
A_s = 2.1e-9
n_s = 0.968

"Maximum scale in units of h/Mpc"
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

'''Create ranges of k and z for the CLASS power spectrum'''
my_k = np.logspace(-3, np.log10(k_max), num=10000) #h/Mpc^-1
my_z_new = np.linspace(0,3,1000)
model = bihalofit(params, my_k, my_z_new)
maximum_distance = 4400
mynz = np.loadtxt("bin_04_desy3_source_nz.dat")
'''input in acrmins'''
my_equilateral_xs = np.logspace(np.log10(3),np.log10(35), num=10)
my_u_values = 1*np.ones_like(my_equilateral_xs)
my_v_values = np.zeros_like(my_equilateral_xs)
limits = [[0, 2*np.pi],[0, np.pi/2],[0,50]]

model.compute_lensing_kernel(1, maximum_distance, 10000, mynz)


'''test reverse order'''

aa_vals = np.ndarray(shape=len(my_equilateral_xs), dtype=complex)
for i in range(len(my_equilateral_xs)):
    test_integration_0_re = model.gamma0_loop(limits, my_equilateral_xs[i], my_u_values[i], my_v_values[i], 100, 4000, 10, imag=False)
    test_integration_0_im = model.gamma0_loop(limits, my_equilateral_xs[i], my_u_values[i], my_v_values[i], 100, 4000, 10, imag=True)
    aa_vals[i] = test_integration_0_re + 1j*test_integration_0_im

result_array_0 = transform_gamma(aa_vals, 0, my_equilateral_xs, my_u_values, my_v_values)
np.save("Gamma0_real_10bins_d2min3_d2max35_NEW_ORDER_10chisteps", np.real(result_array_0))
np.save("Gamma0_imag_10bins_d2min3_d2max35_NEW_ORDER_10chisteps", np.imag(result_array_0))


print(sfbf)
bb_vals = np.ndarray(shape=len(my_equilateral_xs), dtype=complex)
for i in range(len(my_equilateral_xs)):
    test_integration_1_re = model.gamma1(limits, my_equilateral_xs[i], my_u_values[i], my_v_values[i], imag=False)
    test_integration_1_im = model.gamma1(limits, my_equilateral_xs[i], my_u_values[i], my_v_values[i], imag=True)
    bb_vals[i] = test_integration_1_re.mean + 1j*test_integration_1_im.mean

result_array_1 = transform_gamma(bb_vals, 1, my_equilateral_xs, my_u_values, my_v_values)
np.save("Gamma1_real_10bins_d2min3_d2max35_phi60_neval60000_niter8_DEBUG_TAN", np.real(result_array_1))
np.save("Gamma1_imag_10bins_d2min3_d2max35_phi60_neval60000_niter8_DEBUG_TAN", np.imag(result_array_1))

print(sdva)
cc_vals = np.ndarray(shape=len(my_equilateral_xs), dtype=complex)
for i in range(len(my_equilateral_xs)):
    test_integration_2_re = model.gamma2(limits, my_equilateral_xs[i], my_u_values[i], my_v_values[i], imag=False)
    test_integration_2_im = model.gamma2(limits, my_equilateral_xs[i], my_u_values[i], my_v_values[i], imag=True)
    cc_vals[i] = test_integration_2_re.mean + 1j*test_integration_2_im.mean

result_array_2 = transform_gamma(cc_vals, 2, my_equilateral_xs, my_u_values, my_v_values)
np.save("Gamma2_real_10bins_d2min3_d2max35_phi60_neval25000", np.real(result_array_2))
np.save("Gamma2_imag_10bins_d2min3_d2max35_phi60_neval25000", np.imag(result_array_2))

dd_vals = np.ndarray(shape=len(my_equilateral_xs), dtype=complex)
for i in range(len(my_equilateral_xs)):
    test_integration_3_re = model.gamma3(limits, my_equilateral_xs[i], my_u_values[i], my_v_values[i], imag=False)
    test_integration_3_im = model.gamma3(limits, my_equilateral_xs[i], my_u_values[i], my_v_values[i], imag=True)
    dd_vals[i] = test_integration_3_re.mean + 1j*test_integration_3_im.mean

result_array_3 = transform_gamma(dd_vals, 3, my_equilateral_xs, my_u_values, my_v_values)
np.save("Gamma3_real_10bins_d2min3_d2max35_phi60_neval25000", np.real(result_array_3))
np.save("Gamma3_imag_10bins_d2min3_d2max35_phi60_neval25000", np.imag(result_array_3))
