import numpy as np
from astropy.table import Table
from astropy.io import fits
import os
here = os.path.dirname(os.path.abspath(__file__))

def create_full_cov_fits(nsim=400, source='../data/covariance/cov_xip_xim_map3.dat', out='../data/covariance/cov_xip_xim_map3.fits'):
    cov = np.loadtxt(os.path.join(here, source))
    # xip = 200
    # xim = 200
    # map3 = 80
    
    primary = fits.PrimaryHDU()
    hdul = fits.HDUList([primary])
    header = fits.Header()
    header['EXTNAME']  = 'COVMAT'
    header['NSIM']     = nsim
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

    filename = os.path.join(here, out)
    hdul.writeto(filename, overwrite=True)

def rotate_2ptcov_by_analytic_matix():
    """
    See ../plot-notebooks/covarianec-2pt-sim-vs-analytic.ipynb 
    for the idea and details.
    """
    # read analytic covariance
    hdul = fits.open(os.path.join(here,'../data/dv/sim_2pt-NLA.fits'))
    cov_ana = hdul[1].data[:400, :400]
    # compute rotation matrix
    D_ana, R = np.linalg.eig(cov_ana)

    # read simulation-based covariance
    hdul = fits.open(os.path.join(here, '../data/covariance/cov_xip_xim_map3.fits'))
    cov_sim = hdul['COVMAT'].data[:400, :400]

    # Rotate the simulation-based covariance using the rotation matrix obtained above
    D_sim = np.diag((R.T.dot(cov_sim)).dot(R))

    # rescaling the diagonal elements
    f = np.mean(D_ana/D_sim)
    cov_sim_resc = f*(R.dot(np.diag(D_sim))).dot(R.T)

    # write
    hdul['COVMAT'].data[:400, :400] = cov_sim_resc
    hdul.writeto(os.path.join(here, '../data/covariance/cov_xip_xim_map3_rotated.fits'), overwrite=True)

if __name__ == '__main__':
    # create_full_cov_fits()

    #rotate_2ptcov_by_analytic_matix()
    create_full_cov_fits(nsim=300, source='../data/covariance/cov_xip_xim_map3_nreal300.dat', out='../data/covariance/cov_xip_xim_map3_nreal300.fits')
