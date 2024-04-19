"""
Prepare the binning for the map3 signals.

Becasue the third order shear statistics is very computational expensive,
we first fix the combinations of redshift bins and angular bins for which
we want to calculate the third order shear statistics.
"""
from cosmosis.datablock import option_section, names
import numpy as np
import os
from threepoint import ThreePointDataClass

def setup(options):
    config = {}
    fname = options.get_string(option_section, "data_file")
    thpt = ThreePointDataClass.from_fits(fname)
    # Apply selection here
    selection = np.ones(thpt.size, dtype=bool)



    thpt.replace(selection)
    # get binning
    scombs  = thpt.get_z_bin(unique=True).T
    filters = {}
    for scomb in scombs:
        sel = thpt.selection_z_bin(scomb, 'z123', condition='==')
        name= '_'.join([str(s) for s in scomb])
        filters[name] = thpt.get_t_bin(sel=sel)
    # put them on config
    config['sample_combinations'] = scombs
    config['filters'] = filters

    return config

def execute(block, config):
    # map3:
    block['map3', 'sample_combinations'] = config['sample_combinations']
    for scomb in config['sample_combinations']:
        name= '_'.join([str(s) for s in scomb])
        block['map3', 'filters_{}'.format(name)] = config['filters'][name]
    # Because the natural components are also needed for map3
    # we need to inform the z-bin combination to the natural component
    block['natural_components', 'sample_combinations'] = config['sample_combinations']
    return 0

def cleanup(config):
    pass
