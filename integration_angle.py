import time
import numpy as np
import matplotlib.pyplot as plt
from bihalofit import bihalofit
from funcs import f_h3, f_psi3, f_psi1, f_psi2, transform_gamma

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
my_k_reduced = np.logspace(-4, np.log10(k_max), num=1000) #h/Mpc^-1
my_z = np.linspace(0,5,1000)
my_z_simple = np.array([0,0.55])
my_z_new = np.linspace(0,3,1000)
model = bihalofit(params, my_k, my_z_new)

maximum_distance = 4400
mynz = np.loadtxt("bin_04_desy3_source_nz.dat")

plt.plot(mynz[:,0], mynz[:,1])
plt.show()
model.compute_lensing_kernel(1, maximum_distance, 10000, mynz)

'''input in acrmins'''
my_angles = np.linspace(2,178, num=20)
u_2 = np.ones_like(my_angles)
v_2 = np.zeros_like(my_angles)
new_d2 = 21.4*np.ones_like(u_2)
for i in range(len(my_angles)):
    if my_angles[i] < 60:
        u_2[i] = np.sqrt(2*(1-np.cos(my_angles[i]*np.pi/180)))
    else:
        v_2[i] = np.sqrt(2*(1-np.cos(my_angles[i]*np.pi/180)))-1

print(new_d2)
print(u_2)
print(v_2)

limits = [[0, 2*np.pi],[0, np.pi/2],[0,100]]

#from mpi4py import MPI
#def gamma0_mpi(i, model, limits, r, u, v):
#    np.save("Gamma_0_i="+str(i), model.gamma0(limits, r[i], u[i], v[i]))

#run_count = 0
#number_or_runs = 15
#while run_count<number_or_runs:
#                comm = MPI.COMM_WORLD
#                if run_count+comm.rank<number_or_runs:
#                    gamma0_mpi(run_count+comm.rank, model, limits, my_equilateral_xs, my_u_values, my_v_values)
#                run_count+=comm.size
#                comm.bcast(run_count,root = 0)
#                comm.Barrier()

#print(kal)

aa_vals = np.ndarray(shape=len(new_d2), dtype=complex)
for i in range(len(new_d2)):
    test_integration_0_re = model.gamma0(limits, new_d2[i], u_2[i], v_2[i], imag=False)
    test_integration_0_im = model.gamma0(limits, new_d2[i], u_2[i], v_2[i], imag=True)
    aa_vals[i] = test_integration_0_re.mean + 1j*test_integration_0_im.mean

#realp = np.load("/Users/gchgomes/Desktop/3pt_analyses/Gamma0_real_orthocenter_20bins_d2val21point4_phimin0_phimax180_neval_25000_mar30.npy")
#imagp = np.load("/Users/gchgomes/Desktop/3pt_analyses/Gamma0_imag_orthocenter_20bins_d2val21point4_phimin0_phimax180_neval_25000_mar30.npy")
#np.save("Gamma0_real_orthocenter_20bins_d2val21point4_phimin0_phimax180_neval_25000_mar30", np.real(aa_vals))
#np.save("Gamma0_imag_orthocenter_20bins_d2val21point4_phimin0_phimax180_neval_25000_mar30", np.imag(aa_vals))
#aa_vals = realp + 1j*imagp
result_array_0 = transform_gamma(aa_vals, 0, new_d2, u_2, v_2)
np.save("Gamma0_real_20bins_d2val21point4_phimin0_phimax180_neval_60000_niter8_rlim100", np.real(result_array_0))
np.save("Gamma0_imag_20bins_d2val21point4_phimin0_phimax180_neval_60000_niter8_rlim100", np.imag(result_array_0))
print(nn)
test_integration_1 = model.gamma1(limits, my_equilateral_xs, my_u_values, my_v_values)
bb_vals = [test_integration_1[i].mean for i in range(len(test_integration_1))]
result_array_1 = transform_gamma(bb_vals, 1, my_equilateral_xs, my_u_values, my_v_values)
np.save("Gamma1_15bins_d2min3_d2max35_phi60_centroid_no_linterp_no_kinterp", np.real(result_array_1))

test_integration_2 = model.gamma2(limits, my_equilateral_xs, my_u_values, my_v_values)
cc_vals = [test_integration_2[i].mean for i in range(len(test_integration_2))]
result_array_2 = transform_gamma(cc_vals, 2, my_equilateral_xs, my_u_values, my_v_values)
np.save("Gamma2_15bins_d2min3_d2max35_phi60_centroid_no_linterp_no_kinterp", np.real(result_array_2))

test_integration_3 = model.gamma3(limits, my_equilateral_xs, my_u_values, my_v_values)
dd_vals = [test_integration_3[i].mean for i in range(len(test_integration_3))]
result_array_3 = transform_gamma(dd_vals, 3, my_equilateral_xs, my_u_values, my_v_values)
np.save("Gamma3_15bins_d2min3_d2max35_phi60_centroid_no_linterp_no_kinterp", np.real(result_array_3))

result_ttt = (result_array_0+result_array_1+result_array_2+result_array_3)/4