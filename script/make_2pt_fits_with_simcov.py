from astropy.io import fits
import os
import numpy as np
here = os.path.dirname(__file__)

cov_file = os.path.join(here, '../data/covariance/cov_xip_xim_map3.dat')
cov = np.loadtxt(cov_file)[:400, :400]

twopt_file= os.path.join(here, '../data/dv/sim_2pt-NLA-cosmoDESY3.fits')
hdul = fits.open(twopt_file)
hdul['COVMAT'].data[:400,:400] = cov

out_file = os.path.join(here, '../data/dv/sim_2pt-NLA-cosmoDESY3-400sim-cov.fits')
hdul.writeto(out_file, overwrite=True)


## check
import matplotlib.pyplot as plt
hdul1 = fits.open(twopt_file)
hdul2 = fits.open(out_file)
plt.figure(figsize=(15, 5))
plt.yscale('log')
plt.plot(np.diag(hdul1['COVMAT'].data), label='analytic')
plt.plot(np.diag(hdul2['COVMAT'].data), label='400 sims')
plt.legend()
plt.show()
