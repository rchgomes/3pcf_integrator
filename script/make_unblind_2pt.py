"""
This script makes unblinded data vector from the unblind catalog
"""

import os
import sys
from astropy.io import fits
import numpy as np
here = os.path.dirname(__file__)

def make(dirname_2pt, fname_out_2pt):
    """
    dirname_2pt: directory name where the 2pcf measurement data is saved.
    fname_out_2pt: fname to write
    """
    # Read 2pt from measurement file
    xip = []
    xim = []
    for t1 in range(4):
        for t2 in range(4):
            if t1>t2: continue
            hdul = fits.open(os.path.join(dirname_2pt, \
                        f'2PCF_Y3_zbin{t1+1}{t2+1}.fits'))
            xip.append(hdul[1].data['xip'])
            xim.append(hdul[1].data['xim'])
    xip = np.hstack(xip)
    xim = np.hstack(xim)

    # Open a data vector file
    fname_base_2pt  = os.path.join(here, '../data/dv/blind-metacal-2pt-DESY3.fits')
    print(f'Loading base dv file {fname_base_2pt}')
    hdul = fits.open(fname_base_2pt)

    # Write the measured data vector
    # Note 2/3 is the xip/xim hdu
    hdul[2].data['VALUE'] = xip
    hdul[3].data['VALUE'] = xim
    
    # Save the data as a new file
    print(f'Writing to {fname_out_2pt} ...')
    hdul.writeto(fname_out_2pt, overwrite=True)

if __name__ == '__main__':
    ############################################################
    #
    # THIS VERSION WAS STILL DEBUGGING VERSION,
    # NOT RELIABLE!
    if False:
        print('Working on unblinded catalog (latest)')
        # Unblinded catalog
        # (Latest version with v0.5)
        dirname_2pt   = '/global/cfs/cdirs/des/rchgoms1/y3-measurements/2PCF_data_unblinded'
        fname_out_2pt = os.path.join(here, '../data/dv/real_2pt_desy3_unblind_2025Mar9.fits')
        make(dirname_2pt, fname_out_2pt)

    if False:
        print('Working on unblinded catalog (old)')
        # Unblinded catalog 
        # (OLD with sompz v0.4, same as initial public version)
        dirname_2pt   = '/global/cfs/cdirs/des/rchgoms1/y3-measurements/2PCF_data_unblinded_old'
        fname_out_2pt = os.path.join(here, '../data/dv/real_2pt_desy3_unblind_old_2025Mar9.fits')
        make(dirname_2pt, fname_out_2pt)

    if False:
        print('Working on unblinded catalog (latest, g2 flip)')
        # Unblinded catalog
        # (Latest version with v0.5)
        # Debugging of sign problem: g2 -> -g2
        dirname_2pt   = '/global/cfs/cdirs/des/rchgoms1/y3-measurements/2PCF_data_unblinded_g2flip'
        fname_out_2pt = os.path.join(here, '../data/dv/real_2pt_desy3_unblind_g2flip_2025Mar9.fits')
        make(dirname_2pt, fname_out_2pt)

    if False:
        print('Working on unblinded catalog (old, g2 flip)')
        # Unblinded catalog 
        # (OLD with sompz v0.4, same as initial public version)
        # Debugging of sign problem: g2 -> -g2
        dirname_2pt   = '/global/cfs/cdirs/des/rchgoms1/y3-measurements/2PCF_data_unblinded_old_g2flip'
        fname_out_2pt = os.path.join(here, '../data/dv/real_2pt_desy3_unblind_old_g2flip_2025Mar9.fits')
        make(dirname_2pt, fname_out_2pt)

    ############################################################
    #
    # THIS IS THE APPROPRIATE VERSION:
    # We debugged with Elisa and Jamie
    # This datavector completely agree with the public version.
    # The problem was 
    # - bin_slop=0 (we used None)
    # - wpos=ones (we used None)
    # The former makes the main differnece.
    if False:
        print('Working on unblinded catalog (latest, g2 flip)')
        dirname_2pt   = '/global/cfs/cdirs/des/rchgoms1/y3-measurements/2PCF_data_unblinded_g2flip_matchElisa'
        fname_out_2pt = os.path.join(here, '../data/dv/real_2pt_desy3_unblind_g2flip_matchElisa_2025Mar10.fits')
        make(dirname_2pt, fname_out_2pt)

    # Unblidned but with sompzv04 for comaprison to publisehd version
    if True:
        print('Working on unblinded catalog (latest, g2 flip)')
        dirname_2pt   = '/global/cfs/cdirs/des/rchgoms1/y3-measurements/2PCF_data_unblinded_old_g2flip_matchElisa'
        fname_out_2pt = os.path.join(here, '../data/dv/real_2pt_desy3_unblind_g2flip_matchElisa_sompzv04_2025Mar14.fits')
        make(dirname_2pt, fname_out_2pt)

    # Blind
    if False:
        print('Working on blinded catalog')
        # Blinded catalog
        # (Not sure, but this should be blinded catalog that Rafael used.)
        dirname_2pt   = '/global/cfs/cdirs/des/rchgoms1/y3-measurements/2PCF_data_blind'
        fname_out_2pt = os.path.join(here, '../data/dv/real_2pt_desy3_blind_2025Mar9.fits')
        make(dirname_2pt, fname_out_2pt)
    
    if False:
        print('Working on blinded catalog (g2 flip)')
        # Blinded catalog
        # (Not sure, but this should be blinded catalog that Rafael used.)
        dirname_2pt   = '/global/cfs/cdirs/des/rchgoms1/y3-measurements/2PCF_data_blind_g2flip'
        fname_out_2pt = os.path.join(here, '../data/dv/real_2pt_desy3_blind_g2flip_2025Mar9.fits')
        make(dirname_2pt, fname_out_2pt)