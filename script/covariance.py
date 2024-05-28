import numpy as np
from astropy.table import Table
from astropy.io import fits
import os
here = os.path.dirname(os.path.abspath(__file__))

def create_full_cov_fits():
    cov = np.loadtxt(os.path.join(here, '../data/covariance/cov_xip_xim_map3.dat'))
    # xip = 200
    # xim = 200
    # map3 = 80
    
    primary = fits.PrimaryHDU()
    hdul = fits.HDUList([primary])
    header = fits.Header()
    header['EXTNAME']  = 'COVMAT'
    header['NSIM']     = 400
    header['3PT_DATA'] = True
    header['STRT_0'] = 0
    header['name_0'] = 'xip'
    header['STRT_1'] = 200
    header['name_1'] = 'xim'
    header['STRT_2'] = 400
    header['name_2'] = 'map3'
    # create hdu
    hdu = fits.ImageHDU(cov, header=header)
    hdul.append(hdu)

    filename = os.path.join(here, '../data/covariance/cov_xip_xim_map3.fits')
    hdul.writeto(filename, overwrite=True)

if __name__ == '__main__':
    create_full_cov_fits()