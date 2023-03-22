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
mynz = np.loadtxt("bin_04_desy3_source_nz.dat")
model = bihalofit(params,my_k_reduced,z)
model.compute_lensing_kernel(1,4500,1000,mynz)

ell = np.logspace(0,np.log10(4000),1000)
ell_iso0 = 1000*np.ones_like(ell)
ell_iso = 2000*np.ones_like(ell)
ell_iso2 = 3000*np.ones_like(ell)
ell_iso3 = 4000*np.ones_like(ell)
print(ell)
case_1 = model.compute_kappa_bispectrum(ell,ell,ell,4500,500)
case_2 = model.compute_kappa_bispectrum(ell_iso0,ell_iso0,ell/2,4500,500)
case_3 = model.compute_kappa_bispectrum(ell_iso,ell_iso,ell,4500,500)
case_4 = model.compute_kappa_bispectrum(ell_iso2,ell_iso2,ell,4500,500)
case_5 = model.compute_kappa_bispectrum(ell_iso3,ell_iso3,ell,4500,500)
print(case_1)

print(model.compute_kappa_bispectrum(30,30,5000,4500,500))
plt.title("Kappa bispectrum")
plt.plot(ell,case_1, color='b', label='equilateral (B(l,l,l))')
plt.plot(ell/2,case_2, color='darkorange', label='isosceles (B(1000,1000,l))')
plt.plot(ell,case_3, color='r', label='isosceles (B(2000,2000,l))')
plt.plot(ell,case_4, color='g', label='isosceles (B(3000,3000,l))')
plt.plot(ell,case_5, color='k', label='isosceles (B(4000,4000,l))')
plt.legend()
plt.xlim(1,4500)
plt.yscale('log')
plt.xscale('log')
plt.show()
#plt.savefig("Kappa_bispectrum_cases.pdf", dpi=500)

phi = np.linspace(0,2*np.pi,2000)
ell3z = np.sqrt(500**2+500**2+2*500*500*np.cos(phi))
ell1z = 500*np.ones_like(ell3z)
ell3 = np.sqrt(1000**2+1000**2+2*1000*1000*np.cos(phi))
ell1 = 1000*np.ones_like(ell3)
ell3b = np.sqrt(2000**2+2000**2+2*2000*2000*np.cos(phi))
ell1b = 2000*np.ones_like(ell3b)
case_1phi = model.compute_kappa_bispectrum(ell1z,ell1z,ell3z, 5000,500)
case_2phi = model.compute_kappa_bispectrum(ell1,ell1,ell3, 5000,500)
case_3phi = model.compute_kappa_bispectrum(ell1b,ell1b,ell3b, 5000,500)

plt.title("Kappa bispectrum (isosceles)")
plt.plot(180*phi/(np.pi),case_1phi, color='r', label='l1 = 500')
plt.plot(180*phi/(np.pi),case_2phi, color='g', label='l1 = 1000')
plt.plot(180*phi/(np.pi),case_3phi, color='k', label='l1 = 2000')
plt.xlabel("angle")
plt.legend()
plt.yscale('log')
#plt.show()
plt.savefig("Kappa_bispectrum_angles.pdf", dpi=500)
