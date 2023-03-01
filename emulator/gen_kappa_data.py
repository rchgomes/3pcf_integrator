import numpy as np
import pyDOE
import corner
from pathlib import Path
import multiprocessing as mp
from itertools import repeat
import numpy as np
import time

from convergence import get_convergence
def get_convergence_wrapper(args):
    # if 1:
    try:
        return get_convergence(*args)
    except Exception as e:
        print("failed", args[0], e)
        return np.zeros_like(args[1]) + np.nan

num_threads = 16
k_max = 30 #UNITS: h/Mpc

data_dir = "../data/"
subdir = "20230301_test"
plots_dir = f"./plots/{subdir}/"
outdata_dir = f"./training_data/{subdir}/"

Path(plots_dir).mkdir(parents=True, exist_ok=True)
Path(outdata_dir).mkdir(parents=True, exist_ok=True)

# parameter ranges for (Omega_b, Omega_m, h, ln(10^10 A_s), n_s)
ranges = np.array([[0.018, 0.026], [0.2, 0.8], [0.6, 0.9], [1.7, 3.9], [0.8, 1.2]])

# Generate the LHS sample using pyDOE
sample_size = 50 #int(5e5)
lhs_sample = pyDOE.lhs(len(ranges), samples=sample_size, criterion='c')

# Scale the LHS sample to the desired parameter ranges
lhs_cosmology = lhs_sample * (ranges[:,1] - ranges[:,0]) + ranges[:,0]

print(lhs_cosmology.shape)

for lhs in lhs_cosmology:
    print(lhs)

# Plot the samples in a corner plot
fig = corner.corner(lhs_cosmology, labels=[r'$\Omega_b$', r'$\Omega_b$', r'$h$', r'$\ln(10^10 A_s)$', r'$n_s$'], \
                    show_titles=True, title_kwargs={"fontsize": 12}, plot_contours=False)
fig.savefig(plots_dir + 'lhs_corner_plot.pdf')

my_k = np.logspace(-3, np.log10(k_max), num=10000) #h/Mpc^-1
my_k_reduced = np.logspace(-2, np.log10(k_max), num=40) #h/Mpc^-1
my_z_new = np.linspace(0,2.5,100)

xx, yy, zz = np.meshgrid(my_k_reduced, my_k_reduced, my_k_reduced, indexing='ij')
my_kgrid = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

l = np.logspace(2,np.log10(9000), 10)
my_nz = np.loadtxt(data_dir + "bin_04_desy3_source_nz.dat")

start = time.time()

my_pool = mp.Pool(processes=num_threads)

lhs_cosmology[:, 3] = np.exp(lhs_cosmology[:, 3]) / 1e10
args = zip(lhs_cosmology, repeat(l), repeat(my_k), repeat(my_kgrid), \
                   repeat(my_nz), repeat(my_z_new), repeat(my_z_new), repeat(k_max))

p0s = my_pool.map(get_convergence_wrapper, args)

np.save(outdata_dir + "data.npy", p0s)

print("done", time.time() - start)