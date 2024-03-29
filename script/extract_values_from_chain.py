"""
This script extracts the best fit values of model parameters from the chain.

Here best fit values are either of
1. maximum a posteriori parameters
2. maxlike parameters
"""

import numpy as np
import os
import configparser

def get_header(fname):
    """
    get the header from the cosmosis chain file
    """
    with open(fname, 'r') as f:
        header = f.readline().replace('#', '')
    return header

def header_to_names(header):
    """
    convert the header to names
    """
    names = header.split('\t')
    names = [name for name in names if len(name)>1]
    # the last four columns are always prior, like, post, weights.
    names = names[:-4]
    # split into section name and param name
    names = [name.split('--') for name in names]
    return np.array(names)

def get_bestfit_param(chain, statname):
    """
    get the best fit parameter from the chain
    """
    if statname == 'like':
        stats = chain[:, -3]
    elif statname == 'post':
        stats = chain[:, -2]
    else:
        raise ValueError("statname should be either like or post")

    bestfit = chain[np.argmax(stats), :-4]
    return bestfit

def group_section(section_and_names, values):
    """
    group the values by section
    """
    sections = np.unique(section_and_names[:, 0])
    grouped = {}
    for section in sections:
        mask = section_and_names[:, 0] == section
        names = section_and_names[mask, 1]
        values_ = values[mask]
        grouped[section] = dict(zip(names, values_))
    return grouped

def extract_values_from_chain(chain_fname, statname='post'):
    """
    extract the best fit values from the chain
    """
    # read the chain
    chain = np.loadtxt(chain_fname)

    # get the header
    header = get_header(chain_fname)

    # get the names
    names = header_to_names(header)

    # get the best fit values
    bestfit = get_bestfit_param(chain, statname)

    # group by section
    grouped = group_section(names, bestfit)

    return grouped

def print_extracted_values(values):
    # print in ini format
    for section, params in values.items():
        print("[{}]".format(section))
        for param, value in params.items():
            print("{} = {}".format(param, value))
        print()

def fix_values_in_existing_ini_to_bestfit(ini_fname, chain_fname, statname='post'):
    """
    fix the values in the existing ini file to the best fit values
    """
    values = extract_values_from_chain(chain_fname, statname)

    ini = configparser.ConfigParser()
    ini.read(ini_fname)

    for section, params in values.items():
        for param, value in params.items():
            if not ini.has_option(section, param):
                continue
            ini[section][param] = str(value)

    return ini

def print_ini(ini):
    for section in ini.sections():
        print("[{}]".format(section))
        for param in ini[section]:
            print("{} = {}".format(param, ini[section][param]))
        print()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("chain_fname", help="chain file name")
    parser.add_argument("--statname", help="statistic name", default='post')
    parser.add_argument("--ini-fname", help="ini file name", default=None)
    args = parser.parse_args()

    if args.ini_fname is None:
        values = extract_values_from_chain(args.chain_fname, args.statname)
        print_extracted_values(values)
        exit()
    else:
        ini = fix_values_in_existing_ini_to_bestfit(args.ini_fname, args.chain_fname, args.statname)
        print_ini(ini)
        exit()
