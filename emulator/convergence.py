import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import RegularGridInterpolator
from scipy.special import jv
from classy import Class
from pathlib import Path

from bispectrum import bispectrum
from bihalofit import bihalofit, tree_level_bispectrum

np.set_printoptions(precision=3)

def get_convergence(cosmology, l, k, kgrid, k_reduced, nz, z, k_max, maximum_distance=3500, timed=True, verbose=1):
    if timed:
        start = time.time()
    Omega_b, Omega_m, h, A_s, n_s = cosmology
    if verbose == 1:
        print("starting", cosmology)
    params = {'output': 'mPk',
             'non linear': 'halofit',
             'Omega_b': Omega_b,
             'Omega_cdm': Omega_m - Omega_b,
             'h': h,
             'A_s': A_s,
             'n_s': n_s,
             'P_k_max_1/Mpc': k_max*h,
             'z_max_pk': 10.}
    model = bihalofit(params, k, kgrid, z)
    model.compute_all_halo(kgrid)
    model.create_interpolated_bispectrum(kgrid, k_reduced)
    model.compute_lensing_kernel(90, maximum_distance, 10000, nz)
    model.compute_kappa_bispectrum_equilateral(l,maximum_distance,1000)
    if timed:
        print(time.time() - start)
    if verbose == 1:
        print("done", cosmology)
    return model.kappa_bispectrum

if __name__ == "__main__":
    
    Omega_b = 0.05
    Omega_m = 0.308
    h = 0.678
    A_s = 2.1e-9
    n_s = 0.968

    k_max = 30 #UNITS: h/Mpc

    data_dir = "../data/"

    plots_dir = "./plots/convergence_test/"
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    my_k = np.logspace(-3, np.log10(k_max), num=10000) #h/Mpc^-1
    my_k_reduced = np.logspace(-2, np.log10(k_max), num=40) #h/Mpc^-1
    my_z_new = np.linspace(0,2.5,100)

    xx, yy, zz = np.meshgrid(my_k_reduced, my_k_reduced, my_k_reduced, indexing='ij')
    my_kgrid = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    l = np.logspace(2,np.log10(9000), 10)
    mynz = np.loadtxt(data_dir + "bin_04_desy3_source_nz.dat")
    
    kbsp_equi = get_convergence((Omega_b, Omega_m, h, A_s, n_s), \
                               l, my_k, my_kgrid, my_k_reduced, mynz, my_z_new, k_max)

    plt.title("Convergence bispectrum for equilateral triangles")
    plt.xlabel("l")
    plt.ylabel("B(l,l,l)")
    plt.plot(l,kbsp_equi, color='m', ls = '--', label="Bihalofit (kappa)")
    plt.grid()
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(plots_dir+"convergence_bispectrum_nol3.pdf", dpi=300)
    plt.close()
