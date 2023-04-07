import numpy as np
import matplotlib.pyplot as plt
from bispectrum import bispectrum
from scipy.interpolate import RegularGridInterpolator

class halo_model_bispectrum(bispectrum):

    def __init__(self, cosmo_params, k, z, file_prefix):

        bispectrum.__init__(self, cosmo_params, k, z)
        bispec_full = np.ndarray(shape=(len(z), len(k), len(k), len(k)))
        for i in range(len(z)):
            bispec_z = np.load(file_prefix + str(i) + ".npy")
            bispec_reshaped = np.reshape(bispec_z, newshape=(len(k),len(k),len(k)))
            bispec_full[i] = bispec_reshaped
        self.halomodel = RegularGridInterpolator((z, k, k, k), bispec_full, bounds_error = False, fill_value = 0)

    def matter_bispectrum(self, zi, k1,k2,k3):
        return(self.halomodel((zi, k1,k2,k3)))