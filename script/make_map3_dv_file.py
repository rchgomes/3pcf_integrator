"""
Script to make map3 fits file from measurement npy data files.
"""
import numpy as np
from astropy.io import fits
import os
here = os.path.dirname(os.path.abspath(__file__))

def make(base_file, source_file, out_file):
    hdul = fits.open(base_file)

    scombs =[[1,1,1], [1,1,2], [1,1,3], [1,1,4], \
             [1,2,2], [1,2,3], [1,2,4], [1,3,3], \
             [1,3,4], [1,4,4], [2,2,2], [2,2,3], \
             [2,2,4], [2,3,3], [2,3,4], [2,4,4], \
             [3,3,3], [3,3,4], [3,4,4], [4,4,4]]

    inc = 0
    for i, scomb in enumerate(scombs):
        arr = np.load(source_file.format(*scomb))
        hdul[1].data['VALUE'][inc:inc+len(arr)] = arr
        inc += len(arr)
    # check size
    assert inc == len(hdul[1].data['VALUE'])

    hdul.writeto(out_file, overwrite=True)

def main1():
    """
    DES Y3 blind analysis, map3
    using catalog at /global/cfs/cdirs/des/www/y3_cats/Y3_mastercat_03_31_20.h5
    """
    base_file = os.path.join(here, '../data/dv/sim_map3-NLA-cosmoDESY3.fits')
    source_file = '/global/cfs/cdirs/des/rchgoms1/y3-measurements/Map3_data/Y3_blind_MAP3_zbin{}{}{}.npy'
    out_file = os.path.join(here, '../data/dv/real_map3_desy3_blind.fits')
    make(base_file, source_file, out_file)

if __name__ == '__main__':
    main1()