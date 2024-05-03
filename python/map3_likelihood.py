from cosmosis.datablock import option_section, names
import numpy as np
from scipy.interpolate import interp1d
import sys
from fast_map3 import calculateMap3
import time

import pickle

def load_obj(name):
        try:
            with open(name + '.pkl', 'rb') as f:
                return pickle.load(f)
        except:
            with open(name + '.pkl', 'rb') as f:
                return pickle.load(f, encoding='latin1')

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

def setup(options):

    path_chains = options.get_string(option_section, "path_chains", default = "")

    filter_1 = options.get_double_array_1d(option_section, "theta_filter_1")
    filter_2 = options.get_double_array_1d(option_section, "theta_filter_2")
    filter_3 = options.get_double_array_1d(option_section, "theta_filter_3")

    all_filters = np.vstack([filter_1, filter_2, filter_3])

    info = load_obj(path_chains)

    config = dict()
    inv_cov,y_obs= info['inv_cov'],info['y_obs']
    config["inv_cov"] = inv_cov
    config["y_obs"] = y_obs
    config["cov"] = info['cov']
    config["filters"] = all_filters

    # sample combinations
    config['sample-combinations'] = get_sample_sombinations(options)

    config["num_sample"] = len(config['sample-combinations'])
    config["num_aperture"] = len(filter_1)

    return config

def execute(block, config):

    name_likelihood = 'map3_like'

    filters = config["filters"]

    # Get bins for natural component predictions:
    section_nc = 'natural_components'
    phi = block[section_nc, 'phi']
    t1  = block[section_nc, 't1'] * 180*60/np.pi # in arcmin
    t2  = block[section_nc, 't2'] * 180*60/np.pi # in arcmin
    logr_bin_size = np.log(t2[1])-np.log(t2[0])
    phi_bin_size = phi[1]-phi[0]
    phi, t1, t2 = np.meshgrid(phi, t1, t2, indexing='ij')

    zbin_combinations = config["sample-combinations"]
    num_zbin_combinations = config["num_sample"]
    num_aperture_combinations = config["num_aperture"]
    num_all_combinations = num_zbin_combinations*num_aperture_combinations

    y = np.zeros(num_all_combinations)

    for i in range(num_zbin_combinations):

        print(zbin_combinations[i])
        print(zbin_combinations)
        print(num_zbin_combinations)
        if isinstance(zbin_combinations[i], tuple):
            name = '_'.join(zbin_combinations[i])
        else:
            name = str(zbin_combinations[i])
        gamma = block[section_nc, f'real-bin_{name}'] + 1j*block[section_nc, f'imag-bin_{name}']

        y_temp = np.real(calculateMap3(gamma, t2, t1, phi, logr_bin_size, phi_bin_size, filters))
        #y_temp = calculateMap3(gamma, t1, t2, phi, logr_bin_size, phi_bin_size, filters)
        block['map3', f'map3-bin_{name}'] = y_temp
        y[i*num_aperture_combinations:(i+1)*num_aperture_combinations] = y_temp 
    block['map3', 'filters'] = filters

    # likelihood
    w = y-config['y_obs']
    chi2 = np.matmul(w,np.matmul(config['inv_cov'],w))
    block[names.likelihoods, name_likelihood] = -0.5 * np.real(chi2)
    print(y)
    print(config['y_obs'])
    #np.save("theory_19April_map3_new_zbins_los-trial3", y)
    np.save("theory_29Apr_cosmogrid_params_no_IA_debugbranch", y)

    return 0

def cleanup(config):
    pass
