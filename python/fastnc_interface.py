'''
Author     : Sunao Sugiyama
Last edit  : 2024/03/07 18:18:56


TODO:
- include cross-redshift bins
- IA
- allow for user-defined t1 and t2 binning to be consistent with treecorr output. (currently the output is on FFT grid of fastnc)
'''
from cosmosis.datablock import option_section, names
import numpy as np
import os
import fastnc
from astropy.cosmology import wCDM

def setup(options):
    """
    Necessary keys in the ini file:
    - Lmax: maximum multipole
    - Mmax: maximum multipole
    - l12bin: number of bins for l12
    - projection: projection of shear, x, cent, ortho (default: x)
    """
    # config
    config = {}
    option_section = "fastnc"

    # bisppectrum model (halofit)
    config['bispectrum'] = fastnc.bispectrum.BispectrumHalofit()

    # multipole-based 3pcf model (fastnc)
    Lmax   = options.get_int(option_section, "Lmax", default = 30)
    Mmax   = options.get_int(option_section, "Mmax", default = 50)
    config['fastnc'] = fastnc.fastnc.FastNaturalComponents(
        Lmax, Mmax, 
        config_bin={'nell12bin':options.get_int(option_section, "nell12bin", default = 150)}
    )

    # projection
    config['shear-projection'] = options.get_string(option_section, "projection", default = "x")

    # binning
    config['mu']  = np.array([0], dtype=int)
    config['phi-bin'] = np.linspace(0, np.pi, 20)

    return config

def execute(block, config):
    
    # BISPECTRUM ######################################
    # update bispectrum with inputs:
    bs = config['bispectrum']
    # cosmological parameters (fastnc accepts cosmo as astropy format)
    cosmo = wCDM(
                H0=100*block[names.cosmological_parameters, 'h0'], \
                Om0=block[names.cosmological_parameters, 'omega_m'], \
                Ode0=1.0-block[names.cosmological_parameters, 'omega_m'], \
                #meta = {'sigma8':block[names.cosmological_parameters, 'sigma8'], \
                meta = {'sigma8':0.8,  # test\
                        'n':block[names.cosmological_parameters, 'n_s']}
    )
    bs.set_cosmology(cosmo)
    # set source distribution
    # bs.set_source_distribution(
    #     z=block['source-redshift', 'z'], #TODO
    #     dNdz=block['source-redshift', 'dNdz'] #TODO
    # )
    # set lensing kernel
    bs.compute_lensing_kernel()
    # set linear matter power spectrum
    # TODO: For now, we source the power spectrum from file for test.
    # to be replaced with prediction by structure module.
    here = os.path.dirname(__file__)
    k, pklin = np.loadtxt(os.path.join(here,'../data/devl/pklin_P18.dat'), unpack=True)
    z, lgr = np.loadtxt(os.path.join(here,'../data/devl/lgr_P18.dat'), unpack=True)
    bs.set_pklin(
                k, pklin        
                # block['matter_power_lin', 'k_h'], \
                # block['matter_power_lin', 'p_k']
    )
    # set lienar growth rate
    bs.set_lgr(
        z, lgr
        # block[names.cosmological_parameters, 'z'], \
        # block[names.cosmological_parameters, 'growth_rate']
    )
    # update the interpolation.
    bs.interpolate()

    # 3PCF ############################################
    # update fastnc:
    nc = config['fastnc']
    nc.set_bispectrum(bs)
    
    # compute 3PCF
    Gamma = nc.Gamma(
        mu=config['mu'], 
        phi=config['phi-bin'],
        projection=config['shear-projection']
        )
    # Now the Gamma has the shape of (mu.size, phi.size, nc.t1.size, nc.t2.size)
    # We reshape it to one dimensional array
    shape = Gamma.shape
    Gamma = np.reshape(Gamma, -1)
    # We reshape the bin parameters to one dimensional array as well
    mu, phi, t1, t2 = np.meshgrid(config['mu'], config['phi-bin'], nc.t1, nc.t2, indexing='ij')
    mu  = np.reshape(mu, -1)
    phi = np.reshape(phi, -1)
    t1  = np.reshape(t1, -1)
    t2  = np.reshape(t2, -1)

    # write to block
    sctname = "ggg"
    block[sctname, 'Gamma-real'] = Gamma.real
    block[sctname, 'Gamma-imag'] = Gamma.imag
    block[sctname, 'mu'] = mu
    block[sctname, 'phi'] = phi
    block[sctname, 't1'] = t1
    block[sctname, 't2'] = t2
    
    return 0

def cleanup(config):
    pass