from cosmosis.datablock import option_section, names
import numpy as np
import os
import fastnc
from astropy.cosmology import wCDM
from cosmopower import cosmopower_NN
import pickle

def get_string_array_1d(options, section, name):

    o = options.get_string(section, name).split()
    o = [x for x in o if x]
    return o    

def get_sample_sombinations(options, separator=',',):

    o = get_string_array_1d(options, option_section, "sample_combinations")
    o = [tuple(x.split(separator)) for x in o]
    return o

def get_healpix_window_function(nside):
    import healpy as hp
    from scipy.interpolate import interp1d
    w = hp.sphtfunc.pixwin(nside)
    l = np.arange(3*nside)
    fnc = interp1d(l, w, kind='linear', bounds_error=False, fill_value=(1,0))
    window = lambda l1, l2, l3: fnc(l1) * fnc(l2) * fnc(l3)
    return window

def load_obj(name):
    try:
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f, encoding='latin1')

def rescale_params(params, scale):
    params_rescaled = np.zeros_like(params)
    params_rescaled[0] = (params[0] - scale['Omega_m']['small']) / (
                scale['Omega_m']['large'] - scale['Omega_m']['small'])
    params_rescaled[1] = (params[1] - scale['s8']['small']) / (scale['s8']['large'] - scale['s8']['small'])
    params_rescaled[2] = (params[2] - scale['h0']['small']) / (scale['h0']['large'] - scale['h0']['small'])
    params_rescaled[3] = (params[3] - scale['Omega_b']['small']) / (scale['Omega_b']['large'] - scale['Omega_b']['small'])
    params_rescaled[4] = (params[4] - scale['ns']['small']) / (scale['ns']['large'] - scale['ns']['small'])
    return (params_rescaled)


def post_process(array, scale):
    out_array = np.zeros_like(array)
    maxv = scale['large']
    minv = scale['small']

    for i in range(len(array[0])):
        tmp = array[:, i] * (maxv[i] - minv[i]) + minv[i]
        out_array[:, i] = 10 ** (tmp)

    return (out_array)

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
                      'NLA':True, 
                      'multiply_Rb':options.get_bool(option_section, "multiply_Rb", default = False)}
    bs = fastnc.bispectrum.BispectrumHalofit(config_halofit)
    #if options.has_value(option_section, "use-pixwin") and options.get_bool(option_section, "use-pixwin"):
    #    bs.set_window_function(get_healpix_window_function(options.get_int(option_section, "nside")))
    config['bispectrum'] = bs

    # sample combinations
    config['sample-combinations'] = get_sample_sombinations(options)
    print(config['sample-combinations'])
    config["num_sample"] = len(config['sample-combinations'])

    filter_1 = options.get_double_array_1d(option_section, "theta_filter_1")
    filter_2 = options.get_double_array_1d(option_section, "theta_filter_2")
    filter_3 = options.get_double_array_1d(option_section, "theta_filter_3")
    filter_num = len(filter_1)

    zarray = options.get_double_array_1d(option_section, "z_values")

    modes = np.arange(filter_num*len(zarray))
    cp_nn = cosmopower_NN(parameters=['Omega_m', 's8', 'h0', 'Omega_b', 'ns'],
                          modes=modes,
                          n_hidden=[64, 256, 1024, 1024, 512, 256, 128],
                          )

    model_filename = options.get_string(option_section, "model_filename")
    cp_nn.restore(model_filename)

    rescaling_filename = options.get_string(option_section, "rescaling_filename", default="rescaling_for_map3_network.pkl")
    with open(rescaling_filename, 'rb') as file:
        rescaling_values = pickle.load(file)

    config['rescaling_params'] = rescaling_values['params']
    config['rescaling_features'] = rescaling_values['features']
    config["filter_num"] = filter_num
    config['zarray'] = zarray
    config['network'] = cp_nn

    path_chains = options.get_string(option_section, "path_chains", default = "")
    info = load_obj(path_chains)
    inv_cov,y_obs= info['inv_cov'],info['y_obs']
    config["inv_cov"] = inv_cov
    config["y_obs"] = y_obs
    config["cov"] = info['cov']
    
    return config

def execute(block, config):

    name_likelihood = 'emu_map3_like'

    filter_num = config["filter_num"]
    zarray = config['zarray']
    cp_nn = config['network']

    parameters = np.zeros(5)
    parameters[0] = block[names.cosmological_parameters, 'omega_m']
    parameters[1] = block[names.cosmological_parameters, 'S_8']
    parameters[2] = block[names.cosmological_parameters, 'h0']
    parameters[3] = block[names.cosmological_parameters, 'omega_b']
    parameters[4] = block[names.cosmological_parameters, 'n_s']

    params_for_network = rescale_params(parameters, config['rescaling_params'])
    test_params_dict = {'Omega_m': [params_for_network[0]],
                        's8': [params_for_network[1]],
                        'h0': [params_for_network[2]],
                        'Omega_b': [params_for_network[3]],
                        'ns': [params_for_network[4]]}

    predictions = cp_nn.predictions_np(test_params_dict)
    predictions_rescaled = post_process(predictions, config['rescaling_features'])

    predictions_newshape = np.zeros(shape=(len(zarray),filter_num))
    for i in range(filter_num):
        predictions_newshape[:,i] = predictions_rescaled[:,i::filter_num]

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

    bs.compute_kernel()

    sctname = "map3"

    y = []

    for sample_combination in config['sample-combinations']:

        name = sample_combination[0] + sample_combination[1] + sample_combination[2]
        chi = bs.z2chi(zarray)

        z2g0 = bs.z2g_dict[sample_combination[0]]
        z2g1 = bs.z2g_dict[sample_combination[1]]
        z2g2 = bs.z2g_dict[sample_combination[2]]

        weight = z2g0(zarray) * z2g1(zarray) * z2g2(zarray) / chi * (1+zarray)**3

        map3 = np.einsum('ij,i->j',predictions_newshape,weight)
        print('map3shape', np.shape(map3))

        y.append(map3)
        block[sctname, f'map3-bin_{name}'] = map3

    print('shapey', np.shape(y))
    y = np.array(y)
    y_all = y.flatten()

    # likelihood
    w = y_all-config['y_obs']
    chi2 = np.matmul(w,np.matmul(config['inv_cov'],w))
    block[names.likelihoods, name_likelihood] = -0.5 * np.real(chi2)
    print(y)
    np.save("emulated_map3_trial0", y_all)
    print(config['y_obs'])

    return 0

def cleanup(config):
    pass
