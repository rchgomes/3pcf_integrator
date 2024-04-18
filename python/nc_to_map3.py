"""
This module convert natural-component (nc) of shear 3pcf to map3.
"""
from cosmosis.datablock import option_section, names
import numpy as np
from fast_map3 import calculateMap3

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
        o = options.get_double_array_1d(option_section, "sample_combinations")
        return o

def setup(options):
    config = dict()

    filter_1 = options.get_double_array_1d(option_section, "theta_filter_1")
    filter_2 = options.get_double_array_1d(option_section, "theta_filter_2")
    filter_3 = options.get_double_array_1d(option_section, "theta_filter_3")
    config["filters"] = np.vstack([filter_1, filter_2, filter_3])

    # sample combinations
    config['sample-combinations'] = get_sample_sombinations(options)

    return config

def execute(block, config):
    filters = config["filters"]

    # Get bins for natural component predictions:
    section_nc = 'natural_components'
    phi = block[section_nc, 'phi']
    t1  = block[section_nc, 't1'] * 180*60/np.pi # in arcmin
    t2  = block[section_nc, 't2'] * 180*60/np.pi # in arcmin
    logr_bin_size = np.log(t2[1])-np.log(t2[0])
    phi_bin_size = phi[1]-phi[0]
    phi, t1, t2 = np.meshgrid(phi, t1, t2, indexing='ij')

    # convert natural component to map3:
    block['map3', 'filters'] = filters
    for sample_combination in config["sample-combinations"]:
        if isinstance(sample_combination, tuple):
            name = '_'.join(sample_combination)
        else:
            name = str(sample_combination)
        gamma = block[section_nc, f'real-bin_{name}'] + 1j*block[section_nc, f'imag-bin_{name}']
        map3 = calculateMap3(gamma, t2, t1, phi, logr_bin_size, phi_bin_size, filters)
        block['map3', f'map3-bin_{name}'] = map3

    return 0

def cleanup(config):
    pass
