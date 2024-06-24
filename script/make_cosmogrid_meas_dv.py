"""
This script make the 2pt and map3 DV files
using the mean of measuremenets from the cosmogrid sims.
"""
from astropy.io import fits
import numpy as np
import os
here = os.path.dirname(__file__)

def make_2pt():
    # Base two-point DV file
    twopt_file = os.path.join(here, '../data/dv/sim_2pt-NLA-cosmoDESY3.fits')
    hdul = fits.open(twopt_file)

    # covariance 
    cov_file = os.path.join(here, '../data/covariance/cov_xip_xim_map3.dat')
    cov = np.loadtxt(cov_file)[:400, :400]

    # average data vector
    dv = np.loadtxt(os.path.join(here, '../data/covariance/ave_xip_xim_map3.dat'))[:400]

    # write
    hdul['COVMAT'].data[:400,:400] = cov
    hdul['xip'].data['VALUE'] = dv[:200]
    hdul['xim'].data['VALUE'] = dv[200:]

    out_file = os.path.join(here, '../data/dv/meas_2pt-ave-cosmogrid.fits')
    hdul.writeto(out_file, overwrite=True)

def make_map3():
    # Base map3 DV file
    map3_file = os.path.join(here, '../data/dv/sim_map3-NLA-cosmoDESY3.fits')
    hdul = fits.open(map3_file)

    # covariance 
    cov_file = os.path.join(here, '../data/covariance/cov_xip_xim_map3.dat')
    cov = np.loadtxt(cov_file)[400:, 400:]

    # average data vector
    dv = np.loadtxt(os.path.join(here, '../data/covariance/ave_xip_xim_map3.dat'))[400:]

    # write
    hdul['COVMAT'].data = cov
    hdul['map3'].data['VALUE'] = -dv # Note the convension for sign of map3 is different in theory and measurement

    out_file = os.path.join(here, '../data/dv/meas_map3-ave-cosmogrid.fits')
    hdul.writeto(out_file, overwrite=True)

if __name__ == '__main__':
    make_2pt()
    make_map3()