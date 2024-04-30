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

    # Get bins for natural component predictions:
    section_nc = 'natural_components'
    phi = block[section_nc, 'phi']
    t1  = block[section_nc, 't1'] * 180*60/np.pi # in arcmin
    t2  = block[section_nc, 't2'] * 180*60/np.pi # in arcmin
    logr_bin_size = np.log(t2[1])-np.log(t2[0])
    phi_bin_size = phi[1]-phi[0]
    phi, t1, t2 = np.meshgrid(phi, t1, t2, indexing='ij')

    zbin_combinations = config["sample-combinations"]
    num_zbin_combinations = config["num_sample"]
    num_aperture_combinations = config["num_aperture"]
    num_all_combinations = num_zbin_combinations*num_aperture_combinations

    y = np.zeros(num_all_combinations)

    for i in range(num_zbin_combinations):

        print(zbin_combinations[i])
        print(zbin_combinations)
        print(num_zbin_combinations)
        if isinstance(zbin_combinations[i], tuple):
            name = '_'.join(zbin_combinations[i])
        else:
            name = str(zbin_combinations[i])
        gamma = block[section_nc, f'real-bin_{name}'] + 1j*block[section_nc, f'imag-bin_{name}']

        y_temp = np.real(calculateMap3(gamma, t2, t1, phi, logr_bin_size, phi_bin_size, filters))
        #y_temp = calculateMap3(gamma, t1, t2, phi, logr_bin_size, phi_bin_size, filters)
        block['map3', f'map3-bin_{name}'] = y_temp
        y[i*num_aperture_combinations:(i+1)*num_aperture_combinations] = y_temp 
    block['map3', 'filters'] = filters

    # likelihood
    w = y-config['y_obs']
    chi2 = np.matmul(w,np.matmul(config['inv_cov'],w))
    block[names.likelihoods, name_likelihood] = -0.5 * np.real(chi2)
    print(y)
    print(config['y_obs'])
    #np.save("theory_19April_map3_new_zbins_los-trial3", y)
    np.save("theory_29Apr_cosmogrid_params_no_IA", y)

    return 0

def cleanup(config):
    pass
