"""
Returns zero for prior-only chain
"""
from cosmosis.datablock import option_section, names
import numpy as np
import os
from astropy.io import fits
import scipy.linalg
from time import time


def setup(options):
    config = None
    return config

def execute(block, config):
    block[names.likelihoods, f'zero_like'] = 0
    return 0

def cleanup(config):
    pass
