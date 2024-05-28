from cosmosis.datablock import option_section, names
import numpy as np
from threepoint import ThreePointDataClass

def setup(options):
    # name of this likelihood
    name = options.get_string(option_section, "like_name", 'moped')

    # List of likelihood names
    likelihoods = options.get_string(option_section, "likelihoods").split()

    # Load transformation matrix
    # Assumes matrix shape is (data_dim, moped_dim), which is the output of moped_sampler.
    data_file = options.get_string(option_section, "data_file")
    transform_matrix = np.loadtxt(data_file).T

    # Possible option to add:
    # max_n = maximum number of MOPED mode to use
    # moped_index = index of the MOPED mode to use

    config = {"name":name, "likelihoods": likelihoods, "transform_matrix": transform_matrix}
    return config

def execute(block, config):

    # Loop over likelihoods
    full_theo = []
    full_data = []
    for likelihood in config["likelihoods"]:
        # Load data
        data = block[names.data_vector, likelihood+"_data"]
        theo = block[names.data_vector, likelihood+"_theory"]

        # append to full data/theory
        full_data.append(data)
        full_theo.append(theo)

    # Transform data and theory
    full_data = np.hstack(full_data)
    full_theo = np.hstack(full_theo)
    transformed_data = np.dot(config["transform_matrix"], full_data)
    transformed_theo = np.dot(config["transform_matrix"], full_theo)

    # MOPED modes are uncorrelated to each other and normalized
    # by construction, so the covariance matrix is unity.
    chi2 = np.sum((transformed_data - transformed_theo)**2)
    block[names.likelihoods, f'{config["name"]}_like'] = -0.5*chi2
    block[names.data_vector, f'{config["name"]}_chi2'] = chi2
    block[names.data_vector, f'{config["name"]}_data'] = transformed_data
    block[names.data_vector, f'{config["name"]}_theory'] = transformed_theo
    block[names.data_vector, f'{config["name"]}_inverse_covariance'] = np.eye(len(transformed_data))
    block[names.data_vector, f'{config["name"]}_transform_matrix'] = config["transform_matrix"]

    return 0

def cleanup(config):
    pass
