'''
Author     : Sunao Sugiyama
Last edit  : 2024/03/18 12:22:06


TODO:
- IA
- allow for user-defined t1 and t2 binning to be consistent with treecorr output. (currently the output is on FFT grid of fastnc)
- phi binning
'''
from cosmosis.datablock import option_section, names
import numpy as np
import os
import fastnc
from astropy.cosmology import wCDM

def get_string_array_1d(options, section, name):
    # I was not able to utilize
    # CosmoSIS python API 
    # options.get_string_array_1d.
    # This is tentative...
    o = options.get_string(section, name).split()
    o = [x for x in o if x]  # Remove empty strings
    return o    

def get_sample_sombinations(options):
    o = get_string_array_1d(options, option_section, "sample_combinations")
    o = [tuple(x.split(',')) for x in o]
    return o

def get_healpix_window_function(nside):
    import healpy as hp
    from scipy.interpolate import interp1d
    w = hp.sphtfunc.pixwin(1024)
    l = np.arange(3*nside)
    fnc = interp1d(l, w, kind='linear', bounds_error=False, fill_value=(1,0))
    window = lambda l1, l2, l3: fnc(l1) * fnc(l2) * fnc(l3)
    return window

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

    # sample combinations
    config['sample-combinations'] = get_sample_sombinations(options)
    print(config['sample-combinations'])

    # natural component indices
    if options.has_value(option_section, "mu"):
        config['mu'] = options.get_int_array_1d(option_section, "mu")
    else:
        print('Setting detault mu to [0,1,2,3].')
        config['mu'] = [0,1,2,3]

    # binning
    config['phi'] = np.linspace(0, np.pi, 20)

    # bin size in log(theta1) = log(theta2)
    if options.has_value(option_section, "dlnt"):
        config['dlnt'] = options.get_double(option_section, "dlnt")
    else:
        print('Setting detault dlnt to None.')
        config['dlnt'] = None

    # pixel window function
    config['use-pixwin'] = options.get_bool(option_section, "use-pixwin", default = False)
    config['nside'] = options.get_int(option_section, "nside", default = 1024)

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
                meta = {'sigma8':block[names.cosmological_parameters, 'sigma_8'], \
                        'n':block[names.cosmological_parameters, 'n_s']}
    )
    bs.set_cosmology(cosmo)
    # set source distribution
    nzbin = block['nz_source', "nbin"]
    bs.set_source_distribution(
        [block['nz_source', "z"] for _ in range(nzbin)],
        [block['nz_source', "bin_%d" % (i+1)] for i in range(nzbin)],
        ['bin_%d' % (i+1) for i in range(nzbin)]
    )
    # set lensing kernel
    bs.compute_lensing_kernel()
    # set linear matter power spectrum
    bs.set_pklin(
        block[names.matter_power_lin, 'k_h'],
        block[names.matter_power_lin, 'p_k'][0,:]
    )
    # set lienar growth rate
    bs.set_lgr(
        block[names.growth_parameters, "z"],
        block[names.growth_parameters, "d_z"]
    )
    # update the interpolation.
    bs.interpolate()

    # 3PCF ############################################
    nc = config['fastnc']
    sctname = "ggg"

    # pixel window function
    # multiplied on bispectrum in Fourier space
    # This is only needed when compared with simulation
    # with finite resolution.
    # Now, this assumes the resolution is determined by the Nside
    # os healpix.
    if config['use-pixwin']:
        print('Applying pixel window function.')
        window = get_healpix_window_function(config['nside'])
    else:
        window = None

    for sample_combination in config['sample-combinations']:
        print('calculating sample_combination:', sample_combination)
        # set bispectrum
        nc.set_bispectrum(bs, sample_combinations=[sample_combination], window=window)
    
        # compute 3PCF
        Gamma = nc.Gamma(
            mu=config['mu'], 
            phi=config['phi'],
            projection=config['shear-projection'],
            dlnt=config['dlnt'],
            sample_combination=sample_combination,
            )

        # write to block
        # Note that the Gamma has the shape of 
        # (mu.size, phi.size, t1.size, t2.size)
        # where 
        # mu = config['mu']
        # phi = config['phi']
        # t1 = nc.t1
        # t2 = nc.t2
        name = ','.join(sample_combination)
        block[sctname, f'Gamma-real-{name}'] = Gamma.real
        block[sctname, f'Gamma-imag-{name}'] = Gamma.imag
    
    # write common parameters
    block[sctname, 'mu'] = config['mu']
    block[sctname, 'phi'] = config['phi']
    block[sctname, 't1'] = nc.t1
    block[sctname, 't2'] = nc.t2

    return 0

def cleanup(config):
    pass