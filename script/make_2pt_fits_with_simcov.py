from astropy.io import fits
import os
import numpy as np
here = os.path.dirname(__file__)

def make(twopt_file, cov_file=None, out_file=None, plot=False):
    if cov_file is None:
        cov_file = os.path.join(here, '../data/covariance/cov_xip_xim_map3.dat')
    if out_file is None:
        out_file = twopt_file.replace('.fits', '-400sim-cov.fits')

    cov = np.loadtxt(cov_file)[:400, :400]

    hdul = fits.open(twopt_file)
    hdul['COVMAT'].data[:400,:400] = cov

    hdul.writeto(out_file, overwrite=True)

    if plot:
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

def main():
    cov_file = os.path.join(here, '../data/covariance/cov_xip_xim_map3.dat')
    twopt_file= os.path.join(here, '../data/dv/sim_2pt-NLA-cosmoDESY3.fits')
    out_file = os.path.join(here, '../data/dv/sim_2pt-NLA-cosmoDESY3-400sim-cov.fits')
    make(cov_file, twopt_file, out_file, plot=True)

if __name__ == '__main__':
    main()