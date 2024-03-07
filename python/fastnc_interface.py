from cosmosis.datablock import option_section, names
import numpy as np
import fastnc
from astropy.cosmology import wCDM

def setup(options):
    # config
    config = {}

    # bisppectrum model (halofit)
    config['bispectrum'] = fastnc.bispectrum.BispectrumHalofit()

    return config

def execute(block, config):
    """
    Input:
    - cosmological parameters
    - linear power spectrum
    - distance-redshift relation
    Survey input:
    - source redshift bins

    Output:
    - bispectrum model
    - 

    TODO:
    - include cross-redshift bins
    - IA
    """
    
    # update bispectrum with inputs:
    bs = config['bispectrum']
    # cosmological parameters (fastnc accepts cosmo as astropy format)
    cosmo = wCDM(
                H0=100*block[names.cosmological_parameters, 'h0'], \
                Om0=block[names.cosmological_parameters, 'omega_m'], \
                Ode0=1.0-block[names.cosmological_parameters, 'omega_m'], \
                meta = {'sigma8':block[names.cosmological_parameters, 'sigma8'], \
                        'n':block[names.cosmological_parameters, 'n_s']}
                )
    bs.set_cosmology(cosmo)
    # set source distribution
    bs.set_source_distribution(
        z=block[option_section, 'z'],
        dNdz=block[option_section, 'dNdz']
    )

