from cosmosis.datablock import option_section, names
import numpy as np
from threepoint import ThreePointDataClass
from utils import get_sample_combinations

def setup(options):
    config = dict()
    data = ThreePointDataClass.from_fits(options.get_string(option_section, "thpt_file"))
    # apply data selection
    sel = np.zeros(data.size, dtype=bool)
    # zbin cut
    sample_combinations = get_sample_combinations(options)
    for scomb in sample_combinations:
        sel |= data.selection_z_bin(scomb, 'z123')
    # filter cut
    fi


    config['data'] = data
    return config

def execute(block, config):
    data = config['data']
    model= data.copy()

    for z1, z2, z3 in model.get_z_bin(unique=True).T:
        sel = model.selection_z_bin([z1, z2, z3], 'z123')
        t1, t2, t3 = model.get_t_bin(sel)
        # check the order of filter is consistent with the data
        assert np.allclose(np.vstack([t1,t2,t3]), block['map3', 'filter']), \
            'inconsistent filter order between data and model in block.'
        map3 = block['map3', 'map3-bin_{}_{}_{}'.format(z1, z2, z3)]
        model.set_value(z1, z2, z3, t1, t2, t3, map3, where=sel)

    map3_data = thpt.get_signal()
    map3_model= model.get_signal()
    diff = map3_data - map3_model
    icov = data.get_inverse_covariance()
    chi2 = np.matmul(diff, np.matmul(icov, diff))

    block[names.likelihoods, 'map3_like'] = -0.5*chi2

    return 0

def cleanup(config):
    pass
