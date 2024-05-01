from cosmosis.datablock import option_section, names
import numpy as np
from threepoint import ThreePointDataClass

def setup(options):
    config = dict()
    config['data'] = ThreePointDataClass.from_fits(options.get_string(option_section, "data_file"))
    return config

def execute(block, config):
    data = config['data'].copy()
    model= config['data'].copy()

    used = np.zeros(data.size, dtype=bool)
    for scomb in block['map3', 'sample_combinations']:
        name = '_'.join([str(s) for s in scomb])
        # get values
        z1, z2, z3 = scomb
        t1, t2, t3 = block['map3', 'filters_'+name]
        map3 = block['map3', 'map3-bin_'+name]
        # determine where to set the map3
        where = model.where_to_set(z1, z2, z3, t1, t2, t3)
        # assign
        model.set_value(z1, z2, z3, t1, t2, t3, map3, where=where)
        # mark as used
        used[where] = True

    # restrict oursefves to the elements that were used
    model.replace(used)
    data.replace(used)
    
    # compute chi2
    map3_data = data.get_signal()
    map3_model= model.get_signal()
    diff = map3_data - map3_model
    icov = data.get_inverse_covariance()
    chi2 = np.matmul(diff, np.matmul(icov, diff))

    block[names.likelihoods, 'map3_like'] = -0.5*chi2
    block[names.data_vector, 'map3_chi2'] = chi2

    return 0

def cleanup(config):
    pass
