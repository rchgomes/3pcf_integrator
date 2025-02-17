
import numpy as np
import matplotlib.pyplot as plt
from getdist import MCSamples

##################################
# parameter name and label mapping
def get_preset_mapping(names):
    # mcmc related output
    mcmc = {'prior':['prior', 'prior'], \
            'like':['like', 'like'], \
            'post':['post', 'post'], \
            'weight':['weight', 'weight']}
    # DES related model params
    des  = {'cosmological_parameters--omega_m':['om', r'\Omega_{\rm m}'], \
            'cosmological_parameters--s_8': ['s8', r'S_8'], \
            'COSMOLOGICAL_PARAMETERS--SIGMA_8': ['sig8', r'\sigma_8'], \
            'cosmological_parameters--h0': ['h0', 'h_0'], \
            'cosmological_parameters--w': ['w0', r'w_0'], \
            'shear_calibration_parameters--m1': ['m1', r'm_1'], \
            'shear_calibration_parameters--m2': ['m2', r'm_2'], \
            'shear_calibration_parameters--m3': ['m3', r'm_3'], \
            'shear_calibration_parameters--m4': ['m4', r'm_4'], \
            'intrinsic_alignment_parameters--a1':['a1', r'A_1'], \
            'intrinsic_alignment_parameters--a2':['a2', r'A_2'], \
            'intrinsic_alignment_parameters--alpha1': ['alpha1', r'\alpha_1'], \
            'intrinsic_alignment_parameters--alpha2': ['alpha2', r'\alpha_2'], \
            'DATA_VECTOR--2PT_CHI2': ['2pt_chi2', r'\chi^2_{\rm 2pt}'], \
            'DATA_VECTOR--MAP3_CHI2': ['map3_chi2', r'\chi^2_{\rm map3}']}
    # make output
    if isinstance(names, str):
        names = [names]
    mapping = {}
    for name in names:
        if name == 'des':
            mapping |= des
        if name == 'mcmc':
            mapping |= mcmc
    return mapping

##################################
# Utilities of post analysis
def read_cosmosis_param_header(fname, mapping=None):
    # get params from header
    with open(fname, 'r') as f:
        params = f.readline().replace('#','').strip().split()
        labels = [None for p in params]
    # mapping of params and labels
    if mapping is not None:
        labels = [mapping[param][1] if param in mapping else param for param in params]
        params = [mapping[param][0] if param in mapping else param for param in params]
    return params, labels

def select_name(params, take=None):
    if take is not None:
        if isinstance(take, str):
            take = [take]
        if 'prior' not in take: take.append('prior')
        if 'like' not in take: take.append('like')
        if 'post' not in take: take.append('post')
        if 'weight' not in take: take.append('weight')
        index = [i for i, p in enumerate(params) if p in take]
    else:
        index = np.arange(len(params))
    return index

def read_cosmosis_value(filename, mapping=None):
    with open(filename, 'r') as f:
        s_mark = 'START_OF_VALUES_INI'
        e_mark = 'END_OF_VALUES_INI'

        lines = f.readlines()

        for i in range(len(lines)):
            if s_mark in lines[i]:
                i_s = i
            if e_mark in lines[i]:
                i_e = i
                break
        
        lines = lines[i_s+1:i_e]
        
        params = {}
        for line in lines:
            if '[' in line and ']' in line:
                section = line[line.find('[')+1:line.find(']')]
            if '=' in line:
                name, values = line.split('=')
                name = name.replace('##', '').replace(' ', '')
                name = '%s--%s'%(section, name)
                if mapping is not None:
                    name = mapping.get(name, [name, None])[0]
                values = [float(_) for _ in values.split()]
                params[name] = values
    return params

def convert_cosmosis_value_to_range(params):
    ranges = {}
    for name, values in params.items():
        if len(values) == 3:
            ranges[name] = [min(values), max(values)]
    return ranges

def get_cosmological_parameter_mean(fname):
    params, _ = read_cosmosis_param_header(fname, None)
    chain = np.loadtxt(fname)
    weight = chain[:,params.index('weight')]
    means = []
    index = []
    for i, param in enumerate(params):
        if 'cosmological_parameters' in param.lower():
            mean = np.sum(chain[:,i]*weight)/np.sum(weight)
            means.append(mean)
            index.append(i)
    means = np.array(means)
    index = np.array(index)
    return index, means

# weight plot
def plot_weight(chain, params, nlive=500):
    w = chain[:,params.index('weight')]
    plt.figure(figsize=(4,2))
    plt.plot(w[:-nlive]) # remove live points for ease of viewing
    plt.show()

##################################
# mcmc chain reader
def read_cosmosis_mcmc_chain(fname, mapping=None, take=None, blind=True, to_mcsamples=False, fname_mean=None, wplot=False, f_icov=None):
    params, labels = read_cosmosis_param_header(fname, mapping)
    ranges = convert_cosmosis_value_to_range(read_cosmosis_value(fname, mapping))
    chain = np.loadtxt(fname)
    # blind
    if blind:
        fname_mean = fname_mean or fname
        index, means = get_cosmological_parameter_mean(fname_mean)
        print('Blinding ', [params[i] for i in index])
        chain[:, index] -= means[None,:]
        for i in index:
            if params[i] in ranges:
                ranges[params[i]] -= means[i]
            labels[i] = r'\Delta '+labels[i]
    # apply selection
    index = select_name(params, take)
    chain = chain[:, index]
    params= list(np.array(params)[index])
    labels= list(np.array(labels)[index])
    # apply rescaling of samples by the rescaling factor for inverse covariance
    if f_icov is not None:
        reweight_samples_by_icov_rescale_factor(chain, params, f_icov)
    # wplot
    if wplot:
        plot_weight(chain, params)
    if to_mcsamples:
        samples = chain_to_mcsamples(chain, params, labels, ranges=ranges)
        return samples
    else:
        return chain, params, labels, ranges

def read_cosmosis_mcmc_des_chain(fname, mapping=None, take=None, blind=True, to_mcsamples=False, fname_mean=None, wplot=False, f_icov=None):
    if mapping is None:
        mapping = get_preset_mapping('des')
    else:
        mapping |= get_preset_mapping('des')
    return read_cosmosis_mcmc_chain(fname, mapping, take, blind, to_mcsamples, fname_mean, wplot, f_icov)

##################################
# fisher output
def _read_cosmosis_fisher_mu(fname, ndim):
    mu = np.zeros(ndim)
    i = 0
    with open(fname, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if '#mu_{}='.format(i) in line:
                mu[i] = float(line.split('=')[1])
                i += 1
            else:
                continue
            if i>ndim:
                break
    # check
    assert i==ndim, 'i={} while ndim={}'.format(i, ndim)
    
    return mu

def read_cosmosis_fisher(fname, mapping=None, take=None, to_mcsamples=False):
    # read param, label, matrix
    params, labels = read_cosmosis_param_header(fname, mapping)
    ranges = convert_cosmosis_value_to_range(read_cosmosis_value(fname, mapping))
    F = np.loadtxt(fname)
    index = select_name(params, take)
    # read reference model parameter:
    mu = _read_cosmosis_fisher_mu(fname, F.shape[0])
    # apply selection
    mu = mu[index]
    F = F[np.ix_(index, index)]
    params= list(np.array(params)[index])
    labels= list(np.array(labels)[index])
    if to_mcsamples:
        samples = fisher_to_mcsamples(mu, F, params, labels, ranges=ranges)
        return samples
    else:
        return mu, F, params, labels, ranges

def read_cosmosis_fisher_des(fname, mapping=None, take=None, to_mcsamples=False):
    if mapping is None:
        mapping = get_preset_mapping('des')
    else:
        mapping |= get_preset_mapping('des')
    return read_cosmosis_fisher(fname, mapping=mapping, take=take, to_mcsamples=to_mcsamples)

def approximate_range_by_Gauss_in_F(F, params, ranges, scale=1):
    """
    Fisher matrix does not care about the prior range.
    Here, we approximate prior range by Gaussian distribution
    whose width corresponds to range length: sigma=(max-min)/2
    """
    for i, param in enumerate(params):
        if param not in ranges:
            continue
        sigma = scale*(ranges[param][1]-ranges[param][0])/2
        F[i,i]+= 1/sigma**2
    return F

##################################
# getdist interface
def chain_to_mcsamples(chain, params, labels, **kwargs):
    w = chain[:,params.index('weight')]
    samples = MCSamples(samples=chain, names=params, labels=labels, weights=w, **kwargs)
    return samples

def fisher_to_mcsamples(mu, F, params, labels, seed=0, size=5000, **kwargs):
    rng = np.random.default_rng(seed)
    _ = rng.multivariate_normal(mu, np.linalg.inv(F), size=size)
    samples = MCSamples(samples=_, names=params, labels=labels, **kwargs)
    return samples

# utils
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

def reweight_samples_by_icov_rescale_factor(chain, params, f, minw=1e-3):
    """
    Rescale the weights of samples by a correction factor applied 
    to the inverse covariance.

    The factor can be either of Hartlap or Dodelson-Schneider factor,
    or product of them. See e.g. Eq (23) and (24) of 
    https://arxiv.org/pdf/2110.10141

    This function assumes Gaussian likelihood so that the rescaling on
    icov can be equivalent to the rescaling of loglike.
    """
    # Get loglikelihood and weight
    like = chain[:,params.index('like')]
    w    = chain[:,params.index('weight')]

    # We discard the samples that has relative weight smaller than
    # a certain threshold, for which the rewieghting can fail because of the 
    # too much of upweighting. Intuitively, we are removing the samples
    # at the posterior tails.
    # We use 1e-3 as a default choice, but the final result should not 
    # strongly depends on this choice, that one can always check by changing this value.
    sel  = w > w.max()*minw

    # Compute the rescaling factor
    resc = (f-1.0) * like

    # Rescale the posterior
    # here the second term is intended to avoid overflow due to the large value.
    w[sel] *= np.exp(resc[sel] - resc[sel].max())