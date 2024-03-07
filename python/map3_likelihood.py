from cosmosis.datablock import option_section, names
import numpy as np
from scipy.interpolate import interp1d
import sys
from fast_map3 import calculateMap3

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
    filter_1 = options.get_string(option_section, "theta_filter_1", default = "")
    filter_2 = options.get_string(option_section, "theta_filter_2", default = "")
    filter_3 = options.get_string(option_section, "theta_filter_3", default = "")
    all_filters = np.vstack([filter_1, filter_2, filter_3])

    info = load_obj(path_chains)

    config = dict()
    inv_cov,y_obs= info['inv_cov'],info['y_obs']
    config["inv_cov"] = inv_cov
    config["y_obs"] = y_obs
    config["cov"] = info['cov']
    config["filters"] = all_filters

    return config

def execute(block, config):

    name_likelihood = 'map3_like'
    filters = config["filters"]

    '''TO DO: get Gamma^0, Gamma^1, Gamma^2 and Gamma^3 from the datablock in the right format:
    Gamma^0, Gamma^1, Gamma^2, Gamma^3 must be put into the 4xn array three_pt, where 
    three_pt[i] contains the flattened Gamma^i theory. Get also the n-dimensional arrays 
    d2_vals, d3_vals, phi_vals, with the values on the SAS binning. Get the bin size in logr
    and the phi bin size in radians. Use all this on the following function'''

    y = calculateMap3(three_pt,  d2_vals, d3_vals, phi_vals, logr_bin_size, phi_bin_size, filters)

    print(np.shape(config['y_obs']))
    print(np.shape(y))
    w = y-config['y_obs']

    chi2 = np.matmul(w,np.matmul(config['inv_cov'],w))
    block[names.likelihoods, name_likelihood] = -0.5 * np.real(chi2)

    return 0

def cleanup(config):
    pass
