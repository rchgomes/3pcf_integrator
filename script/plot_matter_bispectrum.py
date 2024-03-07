import numpy as np
import matplotlib.pyplot as plt
from bihalofit import bihalofit
from tree_level import tree_level_bispectrum
from gil_marin_bispectrum import gil_marin

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

my_k_reduced = np.logspace(-4, np.log10(k_max), num=1000)
z = np.linspace(0,3,1000)

'''Comparison plot'''
com = np.load("matter_bispectrum_100logkbins.npy")
sem = np.load("matter_bispectrum_100logkbins_new_neff_z0.npy")

halo = np.load("matter_bispectrum_100logkbins_separate_halo_terms.npy")

plt.plot(halo[:,0], halo[:,1], label='one halo')
plt.plot(halo[:,0], halo[:,2], label = 'three halo')
plt.plot(com[:,0], com[:,1], label= 'old halo model')
plt.plot(sem[:,0], sem[:,1], label='new halo model')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid()
plt.show()

plt.title("Equilateral  bihalofit B(k,k,k)")
plt.plot(sem[:,0], sem[:,1], label='new neff')
plt.plot(com[:,0], com[:,1], label='old neff')
plt.xscale('log')
plt.xlabel("k[h/Mpc]")
plt.yscale('log')
plt.ylabel("B(k,k,k) [Mpc/h]^6")
plt.legend()
plt.grid()
#plt.savefig("Bihalofit_equilateral_with_without_baryons.pdf", dpi=500)
plt.show()

fracdiff = (sem[:,1]-com[:,1])/com[:,1]
plt.title("Fractional difference (with/without baryons)")
plt.plot(sem[:,0], fracdiff)
plt.xscale('log')
plt.xlabel("k[h/Mpc]")
plt.ylabel("frac. diff")
plt.ylim(-0.3,0.3)
plt.grid()
#plt.savefig("Fracdiff_baryons.pdf", dpi=500)
plt.show()

newk = np.logspace(-3,np.log10(k_max), num=100)
modelnewk = bihalofit(params,newk,z)
one_halo = modelnewk.compute_one_halo(0, newk, newk, newk)
three_halo = modelnewk.compute_three_halo(0, newk, newk, newk)
resultnewk = modelnewk.matter_bispectrum(1, newk, newk, newk, baryons= False)
treelevelnewk = tree_level_bispectrum(params, newk, z)
resulttreelevelnewk = treelevelnewk.compute_tree_level(0, newk, newk, newk)


gmk = np.logspace(-3,np.log10(k_max), num=5000)
gmk2 = np.logspace(-3,np.log10(k_max), num=400)
gmz = np.linspace(0,3,num=50)
gmbis = gil_marin(params, gmk, gmz)
gmresult = gmbis.matter_bispectrum(0, newk, newk, newk, baryons = False)

plt.title('Bispectra (Equilateral)')
plt.plot(newk, resultnewk, label='bihalofit')
plt.plot(newk, gmresult, label='gil marin')
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.legend()
plt.savefig('Bispectra_equilateral_bih_glm')
#plt.show()

print(sdgha)
my_z_bins = np.linspace(0,3,num=61)
print(my_z_bins)

logsigma8 = modelnewk.logsigma8_interp(my_z_bins)
knl = modelnewk.knl_interp(my_z_bins)
neff = modelnewk.neff_interp(my_z_bins)

myzdata = np.ndarray((61,4))
myzdata[:,0] = my_z_bins
myzdata[:,1] = 1/knl
myzdata[:,2] = neff
myzdata[:,3] = logsigma8

plt.plot(my_z_bins, 1/knl)
plt.grid()
plt.show()

plt.plot(my_z_bins, neff)
plt.grid()
plt.show()

plt.plot(my_z_bins, logsigma8)
plt.grid()
plt.show()

np.save("z_dependent_params_CORRECTED", myzdata)
filedata = np.ndarray((100,3))
filedata[:,0] = newk
filedata[:,1] = one_halo
filedata[:,2] = three_halo
#np.save("matter_bispectrum_100logkbins_separate_halo_terms", filedata)
print(sdga)
#filepl = np.ndarray((100,2))
#filepl[:,0] = newk
#filepl[:,1] = modelnewk.PL((newk, 0))
#np.save("linear_PL_100logkbins", filepl)

#filedata = np.load("matter_bispectrum_100logkbins.npy")
#print(filedata)
plt.plot(newk, resultnewk)
plt.plot(newk, resulttreelevelnewk)
plt.xscale("log")
plt.yscale("log")
plt.show()

#plt.plot(newk, filepl[:,1])
#plt.xscale("log")
#plt.yscale("log")
#plt.show()


print(ksd)
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
plt.savefig("Matter_bispectrum_flattened.pdf", dpi=500)
#plt.show()

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
#plt.savefig("Matter_bispectrum_equilateral_redshift_dependence.pdf", dpi=500)