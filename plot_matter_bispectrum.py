import numpy as np
import matplotlib.pyplot as plt
from bihalofit import bihalofit

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

my_k_reduced = np.logspace(-4, np.log10(k_max), num=1000)
z = np.linspace(0,3,1000)
model = bihalofit(params,my_k_reduced,z)
case_1 = model.matter_bispectrum(0, my_k_reduced, my_k_reduced, my_k_reduced)
case_2 = model.matter_bispectrum(0.5, my_k_reduced, my_k_reduced, my_k_reduced)
case_3 = model.matter_bispectrum(1, my_k_reduced, my_k_reduced, my_k_reduced)
case_4 = model.matter_bispectrum(1.5, my_k_reduced, my_k_reduced, my_k_reduced)
case_5 = model.matter_bispectrum(2, my_k_reduced, my_k_reduced, my_k_reduced)

plt.title("Matter bispectrum, equilateral")
plt.plot(my_k_reduced, case_1, color = 'g', label='z=0.0')
plt.plot(my_k_reduced, case_2, color = 'r', label='z=0.5')
plt.plot(my_k_reduced, case_3, color = 'b', label='z=1.0')
plt.plot(my_k_reduced, case_4, color = 'k', label='z=1.5')
plt.plot(my_k_reduced, case_5, color = 'darkorange', label='z=2.0')
plt.xscale('log')
plt.yscale('log')
plt.xlim(2*10**(-4), 50)
plt.legend()
#plt.savefig("Matter_bispectrum_equilateral.pdf", dpi=500)
plt.show()

case_1fl = model.matter_bispectrum(0, my_k_reduced, my_k_reduced/2, my_k_reduced/2)
case_2fl = model.matter_bispectrum(0.5, my_k_reduced, my_k_reduced/2, my_k_reduced/2)
case_3fl = model.matter_bispectrum(1, my_k_reduced, my_k_reduced/2, my_k_reduced/2)
case_4fl = model.matter_bispectrum(1.5, my_k_reduced, my_k_reduced/2, my_k_reduced/2)
case_5fl = model.matter_bispectrum(2, my_k_reduced, my_k_reduced/2, my_k_reduced/2)

plt.title("Matter bispectrum, flattened")
plt.plot(my_k_reduced, case_1fl, color = 'g', label='z=0.0')
plt.plot(my_k_reduced, case_2fl, color = 'r', label='z=0.5')
plt.plot(my_k_reduced, case_3fl, color = 'b', label='z=1.0')
plt.plot(my_k_reduced, case_4fl, color = 'k', label='z=1.5')
plt.plot(my_k_reduced, case_5fl, color = 'darkorange', label='z=2.0')
plt.xlim(2*10**(-4), 50)
plt.xscale('log')
plt.yscale('log')
plt.legend()
#plt.savefig("Matter_bispectrum_flattened.pdf", dpi=500)
plt.show()

case_1z = model.matter_bispectrum(z, 10**(-2), 10**(-2), 10**(-2))
case_2z = model.matter_bispectrum(z, 10**(-1), 10**(-1), 10**(-1))
case_3z = model.matter_bispectrum(z, 1.0, 1.0, 1.0)
case_4z = model.matter_bispectrum(z, 10.0, 10.0, 10.0)

plt.title("Matter bispectrum, equilateral, redshift dependence")
plt.plot(z, case_1z, color = 'g', label='k=0.01')
plt.plot(z, case_2z, color = 'r', label='k=0.1')
plt.plot(z, case_3z, color = 'b', label='k=1.0')
plt.plot(z, case_4z, color = 'k', label='k=10.0')
plt.yscale('log')
plt.legend()
#plt.show()
plt.savefig("Matter_bispectrum_equilateral_redshift_dependence.pdf", dpi=500)