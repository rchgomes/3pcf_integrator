import numpy as np
import matplotlib.pyplot as plt
from bispectrum import bispectrum
from scipy.interpolate import interp1d
from pyDOE import lhs

lhs_samples = lhs(2, samples=128)

priors = [[0.2,0.4],[1.1e-9, 3.1e-9]]

for i in range(len(lhs_samples[0])):
    lhs_samples[:,i] = lhs_samples[:,i]*(priors[i][1]-priors[i][0]) + priors[i][0]

np.save("lhs_samples_128", lhs_samples)