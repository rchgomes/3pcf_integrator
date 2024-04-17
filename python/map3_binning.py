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
    fname = options.get_string(option_section, "3pt_file")
    thpt = ThreePointDataClass.from_fits(fname)
    scombs = thpt.get_z_bin(unique=True)
    filters= {}
    for scomb in scombs:
        sel = thpt.selection_z_bin(scomb, 'z123', condition='==')
        filters[scomb] = thpt.get_t_bin(sel=sel)
    # Apply selection here

    # put on config
    config['sample_combinations'] = scombs
    config['filters'] = filters

    return config

def execute(block, config):
    # map3:
    block['map3', 'sample_combinations'] = config['sample_combinations']
    for scomb in config['sample_combinations']:
        block['map3', 'filters_'+'_'.join(scomb)] = config['filter'][scomb]
    # Because the natural components are also needed for map3
    # we need to inform the z-bin combination to the natural component
    block['natural_components', 'sample_combinations'] = config['sample_combinations']
    return 0

def cleanup(config):
    pass
