"""
This script makes unblinded data vector file MAP3_FILE.

Update on Mar 11. We previously missed the scale cuts on 3pcf,
which made the resultant map3 larger because of the contributions
below the scale cuts.
"""

import os
import sys
from astropy.io import fits
import numpy as np
here = os.path.dirname(__file__)

# Read map3 from measurement file
# with scale cut on 3pcf
datavector = np.load('/global/cfs/cdirs/des/rchgoms1/y3-measurements/Y3_unblinded_concatenated_map3_scalecut/map3_concatenated_Y3_unblinded.npy')


# Open a data vector file
fname_base_map3 = os.path.join(here, '../data/dv/real_map3_desy3_blind_FEBRUARY2025_796simcov.fits')
print(f'Loading base dv file {fname_base_map3}')
hdul = fits.open(fname_base_map3)

# Write the measured data vector
# Note 1 is the map3 hdu
hdul[1].data['VALUE'] = datavector

# Save the data as a new file
fname_out_map3 = os.path.join(here, '../data/dv/real_map3_desy3_unblind_scalecut3pcf_796simcov_2025Mar11.fits')
print(f'Writing to {fname_out_map3}...')
hdul.writeto(fname_out_map3, overwrite=True)
