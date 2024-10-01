import numpy as np
from astropy.io import fits
import os
here = os.path.dirname(__file__)

def main():
    """
    In the previous measurement on the blind catalog, we did not include the weight og source galaxies. 
    We reran the measurement on the blind catalog with weight properly.
    The data is saved at 
    
        /global/cfs/cdirs/des/rchgoms1/y3-measurements/Map3_data_trial4/
        
    This script replace the map3 data vector in the fits files that we use cosmology inference.
        
        ../data/dv/real_map3_desy3_blind_weightedcov.fits
    """
    # Read base file in which the data and covariance is saved.
    fname_base = os.path.join(here,'../data/dv/real_map3_desy3_blind_weightedcov.fits')
    print('Base file is {}'.format(fname_base))
    data = fits.open(fname_base)
    
    # replace the data vector meaasured with weight in blind catalog
    n = 0 # increment
    fname_dv = '/global/cfs/cdirs/des/rchgoms1/y3-measurements/Map3_data_trial4/Y3_blind_MAP3_zbin{}{}{}.npy'
    for zbin1 in range(1, 4+1):
        for zbin2 in range(zbin1, 4+1):
            for zbin3 in range(zbin2, 4+1):
                print('Replacing {}{}{}'.format(zbin1, zbin2, zbin3))
                print('  Before: {}'.format(data[1].data[n*4:(n+1)*4]['VALUE']))
                dv = np.load(fname_dv.format(zbin1, zbin2, zbin3))
                # Note that the columns correspond to
                # 1. R = filter radius
                # 2. map3 = 3rd aperture mass
                # 3. var(map3)
                data[1].data[n*4:(n+1)*4]['VALUE'] = dv[1,:]
                print('  After: {}'.format(data[1].data[n*4:(n+1)*4]['VALUE']))
                n+=1
    
    # Save as new file
    fname_new = os.path.join(here,'../data/dv/real_map3_desy3_weightedblind_weightedcov.fits')
    print('Writing to {}'.format(fname_new))
    data.writeto(fname_new, overwrite=True)
    
if __name__ == '__main__':
    main()
    