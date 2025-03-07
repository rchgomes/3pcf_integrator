"""
This script makes unblinded data vector file MAP3_FILE.
"""

import os
import sys
from astropy.io import fits
import numpy as np
here = os.path.dirname(__file__)

# fname_base_2pt  = os.path.join(here, '../data/dv/blind-metacal-2pt-DESY3.fits')
# dirname_2pt  = '/global/cfs/cdirs/des/rchgoms1/y3-measurements/2PCF_data_unblinded'
# fname_out_2pt  = os.path.join(here, '../data/dv/unblind_metacal_2pt_DESY3_2025Mar6.fits')


# Read map3 from measurement file
dirname_map3 = '/global/cfs/cdirs/des/rchgoms1/y3-measurements/Map3_data_unblinded'
print(f'Reading map3 measurements from {dirname_map3}')
datavector = []
for t1 in range(4):
    for t2 in range(4):
        for t3 in range(4):
            if t1>t2: continue
            if t2>t3: continue
            # Note that 
            # 0: filter radius
            # 1: map3 values
            # 2: variance of map3
            map3 = np.load(os.path.join(dirname_map3, \
                        f'Y3_MAP3_zbin{t1+1}{t2+1}{t3+1}.npy'))[1]
            datavector.append(map3)
datavector = np.hstack(datavector)

# Open a data vector file
fname_base_map3 = os.path.join(here, '../data/dv/real_map3_desy3_blind_FEBRUARY2025_796simcov.fits')
print(f'Loading base dv file {fname_base_map3}')
hdul = fits.open(fname_base_map3)

# Write the measured data vector
# Note 1 is the map3 hdu
hdul[1].data['VALUE'] = datavector

# Save the data as a new file
fname_out_map3 = os.path.join(here, '../data/dv/real_map3_desy3_unblind_796simcov_2025Mar7.fits')
print(f'Writing to {fname_out_map3}...')
hdul.writeto(fname_out_map3, overwrite=True)
