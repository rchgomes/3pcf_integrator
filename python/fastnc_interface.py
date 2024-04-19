'''
Author     : Sunao Sugiyama
Last edit  : 2024/04/10 17:39:01


TODO:
- IA
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

def get_sample_sombinations(options, separator=',',):
    try:
        o = get_string_array_1d(options, option_section, "sample_combinations")
        o = [tuple(x.split(separator)) for x in o]
        return o
    except:
        # special case for preparing the sample_combinations
        # used for preparing the training data for emulator.
        # We will train for Gamma w/o LoS integration.
        print('<<<<< Special case for preparing the training data for emulator. >>>>>>')
        o = options.get_double_array_1d(option_section, "sample_combinations")
        return o    

def get_healpix_window_function(nside):
    import healpy as hp
    from scipy.interpolate import interp1d
    w = hp.sphtfunc.pixwin(nside)
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

    # Common setups
    Lmax   = options.get_int(option_section, "Lmax", default = 50)
    multipole_type = options.get_string(option_section, "multipole_type", default = "legendre")

    ########################################################################################
    # bisppectrum model (halofit)
    config_halofit = {'Lmax':Lmax, 
                      'multipole_type':multipole_type, 
                      'NLA':options.get_bool(option_section, 'NLA', default=True)} 
    bs = fastnc.bispectrum.BispectrumHalofit(config_halofit)
    if options.has_value(option_section, "use-pixwin") and options.get_bool(option_section, "use-pixwin"):
        bs.set_window_function(get_healpix_window_function(options.get_int(option_section, "nside")))
    config['bispectrum'] = bs

    ########################################################################################    
    # 3PCF model (fastnc)
    t1 = np.logspace(
        np.log10(options.get_double(option_section, "theta-min") * np.pi/(180*60.0)), # radians
        np.log10(options.get_double(option_section, "theta-max") * np.pi/(180*60.0)), # radians
        options.get_int(option_section, "n-theta-bin"))
    t2 = np.logspace(
        np.log10(options.get_double(option_section, "theta-min") * np.pi/(180*60.0)), # radians
        np.log10(options.get_double(option_section, "theta-max") * np.pi/(180*60.0)), # radians
        options.get_int(option_section, "n-theta-bin"))
    phi = np.linspace(
        options.get_double(option_section, "phi-min"), 
        options.get_double(option_section, "phi-max"),
        options.get_int(option_section, "n-phi-bin"))
    config_3pcf = { \
            'Lmax':Lmax, \
            'Mmax':options.get_int(option_section, "Mmax", default = 30), \
            'projection': options.get_string(option_section, "projection", default = "x"), \
            'nfft': options.get_int(option_section, "nfft", default = 150), \
            't1':t1, 
            't2':t2, \
            'phi':phi, \
            'dlnt':options.get_double(option_section, "dlnt", default = None), \
            'mu':options.get_int_array_1d(option_section, "mu") if options.has_value(option_section, "mu") else [0,1,2,3], \
            'multipole_type':multipole_type, \
            'cache':options.get_bool(option_section, 'use_cache', default = False)}
    config['fastnc'] = fastnc.fastnc.FastNaturalComponents(config_3pcf)

    # sample combinations
    config['sample-combinations'] = get_sample_sombinations(options)
    print(config['sample-combinations'])
    
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
                w0 = block[names.cosmological_parameters, 'w'], \
                meta = {'sigma8':block[names.cosmological_parameters, 'sigma_8'], \
                        'n':block[names.cosmological_parameters, 'n_s']}
    )
    bs.set_cosmology(cosmo)
    # Intrinsic parameter
    bs.set_NLA_param({'AIA':block['intrinsic_alignment_parameters', 'a1'], \
            'alphaIA':block['intrinsic_alignment_parameters', 'alpha1'] , \
            'z0':block['intrinsic_alignment_parameters', 'z_piv']})
    # set source distribution
    nzbin = block['nz_source', "nbin"]
    bs.set_source_distribution(
        [block['nz_source', "z"] for _ in range(nzbin)],
        [block['nz_source', "bin_%d" % (i+1)] for i in range(nzbin)],
        ['%d' % (i+1) for i in range(nzbin)]
    )
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
    # set baryon paramter
    if block.has_value('baryon_parameters', 'fb'):
        fb = block['baryon_parameters', 'fb']
        bs.set_baryon_param({'fb': fb})
    # update the interpolation.
    bs.compute_kernel()
    bs.interpolate(scombs=config['sample-combinations'])
    bs.decompose(scombs=config['sample-combinations'])

    # 3PCF ############################################
    nc = config['fastnc']
    sctname = "natural_components"

    for sample_combination in config['sample-combinations']:
        print('calculating sample_combination:', sample_combination)
        # set bispectrum
        nc.set_bispectrum(bs)
        # nc.set_grid()
        nc.compute(scomb=sample_combination)

        # stack the Gamma
        Gamma = np.array([nc.Gamma0, nc.Gamma1, nc.Gamma2, nc.Gamma3])

        # write to block
        # Note that the Gamma has the shape of 
        # (mu.size, phi.size, t1.size, t2.size)
        if isinstance(sample_combination, tuple):
            name = '_'.join(sample_combination)
        else:
            name = str(sample_combination)
        block[sctname, f'real-bin_{name}'] = Gamma.real
        block[sctname, f'imag-bin_{name}'] = Gamma.imag
    
    # write common parameters
    block[sctname, 'mu'] = nc.mu
    block[sctname, 'phi'] = nc.phi
    # nc.t1 and nc.t2 is the lower edges of bins
    # block[sctname, 't1'] = nc.t1
    # block[sctname, 't2'] = nc.t2
    # Conversion of Gamma to map3 in measurement
    # uses mean t1 and mean t2 as bin values 
    # (meand2, meand3 in TreeCorr)
    # (The other option is to use exp(logmeand2) etc)
    dlnt = np.diff(np.log(nc.t1))[0]
    # 1. meant1 = t1min * 2/3 (exp(3dlnt)-1)/(exp(2dlnt)-1)
    factor = 2.0/3.0*(np.exp(3*dlnt)-1)/(np.exp(2*dlnt)-1)
    block[sctname, 't1'] = nc.t1 * factor
    block[sctname, 't2'] = nc.t2 * factor
    # 2. exp(logmeant1) = t1min * exp( (exp(2dlnt)(2dlnt-1)+1)/2/(exp(2dlnt)-1) )
    # factor = (np.exp(2*dlnt)*(2*dlnt-1)+1)/2/(np.exp(2*dlnt)-1)
    # factor = np.exp(factor)
    # block[sctname, 'meant1'] = nc.t1 * factor
    # block[sctname, 'meant2'] = nc.t2 * factor

    return 0

def cleanup(config):
    pass
