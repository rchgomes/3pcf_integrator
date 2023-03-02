import numpy as np
import pyDOE
import corner
from pathlib import Path
import multiprocessing as mp
from itertools import repeat, count
import numpy as np
import time, shutil, os

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

from convergence import get_convergence
def get_convergence_wrapper(args):
    # if 1:
    try:
        print(args[0])
        return get_convergence(*args[1:])
    except Exception as e:
        print("failed", args[1], e)
        return np.zeros_like(args[2]) + np.nan

num_threads = 32
k_max = 30 #UNITS: h/Mpc


# choose chunk size correctly. next chunk would not start processing till the last one is finished. 
# chunks are supposed to be used (>1) for when timeout on NERSC becomes an issue
# in that case, you could skip previously computed and saved chunks
chunks = 6
l_bins = 10
sample_size = 50 #int(5e5)
hard_reset = False
    
data_dir = "../data/"
subdir = f"20230301_test_s{sample_size}_l{l_bins}_c{chunks}"
plots_dir = f"./plots/{subdir}/"
outdata_dir = f"./training_data/{subdir}/"

if hard_reset and os.path.exists(plots_dir) and os.path.isdir(plots_dir):
    print("deleting old plots subdirectories if they exist")
    shutil.rmtree(plots_dir)
    shutil.rmtree(outdata_dir)
    
if hard_reset and os.path.exists(outdata_dir) and os.path.isdir(outdata_dir):
    print("deleting old outdata subdirectories if they exist")
    shutil.rmtree(outdata_dir)
    
Path(plots_dir).mkdir(parents=True, exist_ok=True)
Path(outdata_dir).mkdir(parents=True, exist_ok=True)

# parameter ranges for (Omega_b, Omega_m, h, ln(10^10 A_s), n_s)
ranges = np.array([[0.01, 0.05], [0.1, 0.5], [0.6, 0.9], [1.7, 3.9], [0.8, 1.2]])
    

try:
    print("reading premade cosmology")
    lhs_cosmology = np.load(outdata_dir + f"cosmology.npy")
except:
    print("no premade cosmology")
    # Generate the LHS sample using pyDOE
    lhs_sample = pyDOE.lhs(len(ranges), samples=sample_size, criterion='c')

    # Scale the LHS sample to the desired parameter ranges
    lhs_cosmology = lhs_sample * (ranges[:,1] - ranges[:,0]) + ranges[:,0]

    # Plot the samples in a corner plot
    fig = corner.corner(lhs_cosmology, labels=[r'$\Omega_b$', r'$\Omega_b$', r'$h$', r'$\ln(10^10 A_s)$', r'$n_s$'], \
                        show_titles=True, title_kwargs={"fontsize": 12}, plot_contours=False)
    fig.savefig(plots_dir + 'lhs_corner_plot.pdf')

    lhs_cosmology[:, 3] = np.exp(lhs_cosmology[:, 3]) / 1e10 # As

    print(lhs_cosmology.shape)
    
    np.save(outdata_dir + f"cosmology.npy", lhs_cosmology)

try:
    print("reading premade l values")
    l = np.load(outdata_dir + f"l.npy")
except:
    print("no premade l values")
    l = np.logspace(2,np.log10(9000), l_bins)
    np.save(outdata_dir + f"l.npy", l)

my_k = np.logspace(-3, np.log10(k_max), num=10000) #h/Mpc^-1
my_k_reduced = np.logspace(-2, np.log10(k_max), num=40) #h/Mpc^-1
my_z_new = np.linspace(0,2.5,100)
xx, yy, zz = np.meshgrid(my_k_reduced, my_k_reduced, my_k_reduced, indexing='ij')
my_kgrid = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
my_nz = np.loadtxt(data_dir + "bin_04_desy3_source_nz.dat")

start = time.time()

my_pool = mp.Pool(processes=num_threads)

cosmology_splits = list(split(lhs_cosmology, chunks))

for i in range(chunks):
    print("chunk", i)
    try:
        _ = np.load(outdata_dir + f"data{i}.npy")
        print("chunk exists, yay")
    except:
        print("chunk does not exists")
        args = zip(count(), cosmology_splits[i], repeat(l), repeat(my_k), repeat(my_kgrid), \
                       repeat(my_k_reduced), repeat(my_nz), repeat(my_z_new), repeat(k_max))
        p0s = my_pool.map(get_convergence_wrapper, args)
        np.save(outdata_dir + f"data{i}.npy", p0s)
        print("done chunk", i)

print("done", time.time() - start)