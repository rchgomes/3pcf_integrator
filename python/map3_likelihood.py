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

    return config

def execute(block, config):

    name_likelihood = 'map3_like'
    filters = config["filters"]

    gamma = block["ggg", 'Gamma-real'] + 1j*block["ggg", 'Gamma-imag']
    mu = block["ggg", 'mu']

    indices_0 = np.where(mu == 0)
    indices_1 = np.where(mu == 1)
    indices_2 = np.where(mu == 2)
    indices_3 = np.where(mu == 3)

    gamma0 = gamma[indices_0]
    gamma1 = gamma[indices_1]
    gamma2 = gamma[indices_2]
    gamma3 = gamma[indices_3]

    phi = block["ggg", 'phi'][indices_0]
    t1 = block["ggg", 't1'][indices_0]
    t2 = block["ggg", 't2'][indices_0]

    t1 *= 180*60/np.pi
    t2 *= 180*60/np.pi

    gamma_all = np.vstack([gamma0, gamma1, gamma2, gamma3])

    '''TO DO: Get the bin size in logr
    and the phi bin size in radians. Use all this on the following function.
    Also, compute for all z auto-correlations'''

    phi_bin_size = np.pi/20
    logr_bin_size = 0.2

    y = calculateMap3(gamma_all,  t2, t1, phi, logr_bin_size, phi_bin_size, filters)

    w = y-config['y_obs']

    chi2 = np.matmul(w,np.matmul(config['inv_cov'],w))
    block[names.likelihoods, name_likelihood] = -0.5 * np.real(chi2)

    return 0

def cleanup(config):
    pass
