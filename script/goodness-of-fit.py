"""
This script calls the different sampler used in the config file, still using the same pipeline.
Especially use the bestfit param of the mcmc chain.

This is useful to assess the goodness-of-fit in the blind analysis,
because this script never leaves value file as a file, whose values must be blinded.

Based on Raveri & Hu (2019), we compute the goodness-of-fit by chi^2/dof, where dof is the effective dof.
The effective dof is computed by the formula:
    Neff = Nparam - Tr(C^-1_prior C_post)
where C_prior and C_post are the covariance matrix of the prior and posterior, respectively.

Usage:
    1. run this script with `--sampler=apriori`
    2. run this script with `--sampler=sample-fisher`
    3. run this script with `--sampler=dof`
    4. run this script with `--sampler=gof`

Example:
    If you want to compute the dof of the analysis with
    config/test-desy3c-joint-2pt-moped-run/des-y3-shear-2pt-map3-NLA-all.ini
    Then 

    1. python script/goodness-of-fit.py config/test-desy3c-joint-2pt-moped-run/des-y3-shear-2pt-map3-NLA-all.ini --sampler=apriori --nsample=1000

    2. python script/goodness-of-fit.py config/test-desy3c-joint-2pt-moped-run/des-y3-shear-2pt-map3-NLA-all.ini --sampler=sample-fisher --nsample=1000

    3. python script/goodness-of-fit.py config/test-desy3c-joint-2pt-moped-run/des-y3-shear-2pt-map3-NLA-all.ini --sampler=dof

    4. python script/goodness-of-fit.py config/test-desy3c-joint-2pt-moped-run/des-y3-shear-2pt-map3-NLA-all.ini --sampler=gof --chi2name=joint_2pt_moped
"""
import numpy as np
from cosmosis import Inifile
from cosmosis.main import run_cosmosis
import argparse
import os

def get_bestfit_values_ini(ini_file):
    if isinstance(ini_file, str):
        ini = Inifile(ini_file, print_include_messages=False)
    else:
        ini = ini_file

    chain_file = ini['output', 'filename']
    with open(chain_file, 'r') as f:
        params = f.readline().replace('#','').strip().split()
    samples = np.loadtxt(chain_file, comments='#')
    lnpost = samples[:, params.index('post')]
    bestfit = samples[np.argmax(lnpost),:]

    value_file = ini['pipeline', 'values']
    ini_value = Inifile(value_file)

    for section in ini_value.sections():
        for option in ini_value.options(section):
            key = '%s--%s'%(section, option)
            if key in params:
                val = ini_value[section, option]
                val = [_ for _ in val.split()]
                val[1] = str(bestfit[params.index(key)])
                ini_value[section, option] = ' '.join(val)

    return ini_value

def get_priors_ini(ini_file):
    if isinstance(ini_file, str):
        ini = Inifile(ini_file, print_include_messages=False)
    else:
        ini = ini_file

    prior_file = ini['pipeline', 'priors']
    ini_prior = Inifile(prior_file)

    return ini_prior

def make_fisher_sampler_ini(ini_file):
    ini = Inifile(ini_file, print_include_messages=False)
    ini['runtime', 'sampler'] = 'fisher'
    ini['output', 'filename'] = ini['output', 'filename'].replace('.txt', '-fisher.txt')
    return ini

def make_apriori_sampler_ini(ini_file, nsample=1000):
    ini = Inifile(ini_file, print_include_messages=False)
    ini['runtime', 'sampler'] = 'apriori'
    # reset pipeline
    ini.remove_section('pipeline')
    ini.add_section('pipeline')
    # apriori section
    ini.add_section('apriori')
    ini['apriori', 'nsample'] = nsample
    ini['output', 'filename'] = ini['output', 'filename'].replace('.txt', '-apriori.txt')
    return ini

def run_fisher(ini_file):
    ini = Inifile(ini_file)
    ini_value = get_bestfit_values_ini(ini)
    ini_prior = get_priors_ini(ini)
    ini_fisher = make_fisher_sampler_ini(ini_file)
    run_cosmosis(ini_fisher, values=ini_value, priors=ini_prior)

def sample_fisher(ini_file, nsample=1000):
    # run fisher
    run_fisher(ini_file)

    print('Start sampling')
    # using the fisher matrix, we sample
    ini = Inifile(ini_file)
    ini_value = get_bestfit_values_ini(ini)
    ini_fisher = make_fisher_sampler_ini(ini_file)

    fisher_file = ini_fisher['output', 'filename']
    priors_file = ini['pipeline', 'priors']
    values_file = 'values-temp.ini'
    output_file = fisher_file.replace('.txt', '-sample.txt')

    # 1. save value file temporarily
    with open(values_file, 'w') as f:
        ini_value.write(f)

    # 2. sample fisher
    command = f'cosmosis-sample-fisher {fisher_file} {values_file} {priors_file} {nsample} {output_file}'
    os.system(command)
    print('output = %s'%output_file)

    # 3. add header
    with open(fisher_file, 'r') as f:
        header = f.readline().replace('#','').strip()
    np.savetxt(output_file, np.loadtxt(output_file), header=header)

    # 4. clean up
    os.remove(values_file)

    # 5. remove the fisher matrix
    os.remove(fisher_file)

def sample_apriori(ini_file, nsamples=1000):
    ini_value = get_bestfit_values_ini(ini_file)
    ini_prior = get_priors_ini(ini_file)
    ini_apriori = make_apriori_sampler_ini(ini_file, nsample=nsamples)
    run_cosmosis(ini_apriori, values=ini_value, priors=ini_prior)
    print('output = %s'%ini_apriori['output', 'filename'])

def compute_effective_dof(ini_file):
    ini_apriori = make_apriori_sampler_ini(ini_file)
    ini_fisher  = make_fisher_sampler_ini(ini_file)

    apriori_file = ini_apriori['output', 'filename']
    fisher_file  = ini_fisher['output', 'filename']
    fisher_sample_file = fisher_file.replace('.txt', '-sample.txt')

    samples_fisher  = np.loadtxt(fisher_sample_file)
    Nparam = samples_fisher.shape[1]
    samples_apriori = np.loadtxt(apriori_file)[:,:Nparam]

    c_prio = np.cov(samples_apriori, rowvar=False)
    c_post = np.cov(samples_fisher, rowvar=False)

    Neff = Nparam - np.trace(np.linalg.inv(c_prio) @ c_post)

    print(f'Nparam = {Nparam}, Neff = {Neff}')
    return Nparam, Neff

def compute_chi2(ini_file, chi2name):
    ini = Inifile(ini_file, print_include_messages=False)
    ini['runtime', 'sampler'] = 'test'
    ini['test', 'save_dir'] = ini['test', 'save_dir']+'-test'
    ini_value = get_bestfit_values_ini(ini_file)
    ini_prior = get_priors_ini(ini_file)
    run_cosmosis(ini, values=ini_value, priors=ini_prior)
    print('output = %s'%ini['test', 'save_dir'])
    # take chi2
    filename = ini['test', 'save_dir']+'/data_vector/values.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if chi2name in line:
                chi2 = float(line.split('=')[1])
                break
    print('======================================================')
    print('Remove the test directory :', ini['test', 'save_dir'])
    print('because this directory contains the blinded values')
    print('======================================================')
    return chi2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inifile', type=str)
    parser.add_argument('--sampler', type=str, default='')
    parser.add_argument('--nsample', type=int, default=1000)
    parser.add_argument('--chi2name', type=str, default='chi2', help='Name of the chi2 in the data_vector')
    args = parser.parse_args()

    if args.sampler == 'fisher':
        # This is just to run fisher
        run_fisher(args.inifile)
    elif args.sampler == 'sample-fisher':
        # This is 1. run fisher then 2. sample fisher
        # This accounts for the prior boundary cut.
        sample_fisher(args.inifile, nsample=args.nsample)
    elif args.sampler == 'apriori':
        sample_apriori(args.inifile, nsamples=args.nsample)
    elif args.sampler == 'dof':
        compute_effective_dof(args.inifile)
    elif args.sampler == 'gof':
        _, Neff = compute_effective_dof(args.inifile)
        chi2 = compute_chi2(args.inifile, args.chi2name)
        print(f'Goodness-of-fit = {chi2}/{Neff} = {chi2/Neff}')
    else:
        raise ValueError(f'Unknown sampler: {args.sampler}')