# This script creates a synthetic map3 data
# from the output of cosmosis with test mode

# Purpose: To create a noiseless data vector
# computed from desired model parameters. Useful for
# check of pipeline

# How to use
# 1. run config/des-y3-shear-2pt-map3-s8om.ini with test mode
# 2. run this script

import numpy as np
import pickle

def load_obj(name):
    try:
        with open(name, 'rb') as f:
            return pickle.load(f)
    except:
        with open(name, 'rb') as f:
            return pickle.load(f, encoding='latin1')

def read(dirname, scombs):
    data = []
    for scomb in scombs:
        _ = np.loadtxt("{}/map-bin_{},bin_{},bin_{}.txt".format(dirname, scomb[0],scomb[1],scomb[2]))
        data.append(_)
    data = np.hstack(data)
    return data

def write(fname_base, fname_out, data):
    """
    Write  the computed synthetic data

    For now we assume Rafael's data format for CosmoGrid sims.
    """
    obj = load_obj(fname_base)
    obj['y_obs'] = data
    with open(fname_out, 'wb') as f:
        pickle.dump(obj, f)

if __name__ == '__main__':
    data = read("output/des-y3-2pt-map3/map3", [(1,1,1), (2,2,2), (3,3,3), (4,4,4)])
    fname_base = "data/nz/map3_dv_COSMOGRID_7Mar24_REDSHIFT_AUTOCORRELATIONS_ONLY.pkl"
    fname_out  = "data/nz/map3_synthetic_desy3-param_REDSHIFT_AUTOCORRELATIONS_ONLY.pkl"
    form = write(fname_base, fname_out, data)
