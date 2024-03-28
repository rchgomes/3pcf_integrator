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
    config['sample-combinations'] = options.get_string(option_section, "sample_combinations", default = "all")
    config['sample-combinations'] = config['sample-combinations'].split(' ')
    config['sample-combinations'] = [tuple(x.split(',')) for x in config['sample-combinations']]

    config["num_sample"] = len(config['sample-combinations'])
    config["num_aperture"] = len(filter_1)

    return config

def execute(block, config):

    name_likelihood = 'map3_like'

    filters = config["filters"]
    phi = block["ggg", 'phi']
    t1 = block["ggg", 't1'] * 180*60/np.pi # in arcmin
    t2 = block["ggg", 't2'] * 180*60/np.pi # in arcmin

    '''TO DO: Get the bin size in logr from input.
    Temporarily, I'm taking the bin size directly from the t2 array values,
    I'm assuming the phi bin size will remain hard coded, but we could put 
    it as an input for fastnc. In this case, we would also change the 
    phi_bin_size parameter here.'''

    logr_bin_size = np.log(t2[1])-np.log(t2[0])
    phi_bin_size = phi[1]-phi[0]

    '''END OF TO DO'''

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
        name = ','.join(zbin_combinations[i])
        gamma = block["ggg", f'Gamma-real-{name}'] + 1j*block["ggg", f'Gamma-imag-{name}']

        y_temp = calculateMap3(gamma, t2, t1, phi, logr_bin_size, phi_bin_size, filters)
        #y_temp = calculateMap3(gamma, t1, t2, phi, logr_bin_size, phi_bin_size, filters)
        block['map3', f'map-bin_{i+1},bin_{i+1},bin_{i+1}'] = y_temp
        y[i*num_aperture_combinations:(i+1)*num_aperture_combinations] = y_temp 
    block['map3', 'filters'] = filters

    # likelihood
    w = y-config['y_obs']
    chi2 = np.matmul(w,np.matmul(config['inv_cov'],w))
    block[names.likelihoods, name_likelihood] = -0.5 * np.real(chi2)
    print(y)
    print(config['y_obs'])
    #np.save("theory_test_autocorrelations_map3", y)

    return 0

def cleanup(config):
    pass
