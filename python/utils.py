import numpy as np
import matplotlib.pyplot as plt
from getdist import MCSamples

# Utilities of post analysis
def read_cosmosis_chain(fname, mapping=None):
    """
    fname (str): path to the chain file
    mapping (dict): mapping of parameter names
    """
    # get params from header
    with open(fname, 'r') as f:
        params = f.readline().replace('#','').strip().split()
        labels = [None for p in params]
        index  = np.arange(len(params))
    # if mapping is provided, use it
    if mapping is not None:
        mapping['prior'] = mapping.get('prior', ['prior', 'prior'])
        mapping['like']  = mapping.get('like', ['like', 'like'])
        mapping['post']  = mapping.get('post', ['post', 'post'])
        mapping['weight']= mapping.get('weight', ['weight', 'weight'])
        index  = [i for i, p in zip(index, params) if p in mapping]
        labels = [mapping[p][1] for p in params if p in mapping]
        params = [mapping[p][0] for p in params if p in mapping]
    # get chain
    chain = np.loadtxt(fname)[:,index]
    return chain, params, labels

def preduce(chain, params, labels, take=None):
    if take is not None:
        if isinstance(take, str):
            take = [take]
        if 'prior' not in take: take.append('prior')
        if 'like' not in take: take.append('like')
        if 'post' not in take: take.append('post')
        if 'weight' not in take: take.append('weight')
        index = [i for i, p in enumerate(params) if p in take]
        chain = chain[:,index]
        params= [params[i] for i in index]
        labels= [labels[i] for i in index]
    return chain, params, labels

def read_des_chain(fname, take=None):
    mapping = {'cosmological_parameters--omega_m':['om', r'\Omega_{\rm m}'], \
        'cosmological_parameters--s_8': ['s8', r'S_8'], \
        'COSMOLOGICAL_PARAMETERS--SIGMA_8': ['sig8', r'\sigma_8'], \
        'cosmological_parameters--w': ['w0', r'w_0'], \
        'shear_calibration_parameters--m1': ['m1', r'm_1'], \
        'shear_calibration_parameters--m2': ['m2', r'm_2'], \
        'shear_calibration_parameters--m3': ['m3', r'm_3'], \
        'shear_calibration_parameters--m4': ['m4', r'm_4'], \
        'intrinsic_alignment_parameters--a1':['a1', r'A_1'], \
        'intrinsic_alignment_parameters--a2':['a2', r'A_2'], \
        'intrinsic_alignment_parameters--alpha1': ['alpha1', r'\alpha_1'], \
        'intrinsic_alignment_parameters--alpha2': ['alpha2', r'\alpha_2']}
    chain, params, labels = read_cosmosis_chain(fname, mapping)
    print(params)
    chain, params, labels = preduce(chain, params, labels, take)
    return chain, params, labels

def to_mcsamples(chain, params, labels):
    w = chain[:,params.index('weight')]
    samples = MCSamples(samples=chain, names=params, labels=labels, weights=w)
    return samples

def wplot(chain, params, labels):
    w = chain[:,params.index('weight')]
    plt.figure(figsize=(4,2))
    plt.plot(w[:-500]) # remove live points for ease of viewing
    plt.show()

def cov_from_samples_names(samples, names):
    pars = [samples.getParamNames().list().index(name) for name in names]
    cov = samples.getCov(pars=pars)
    return cov

def FoM_from_samples_names(samples, names):
    cov = cov_from_samples_names(samples, names)
    fom = np.linalg.det(cov)**-0.5
    return fom
