import numpy as np
import pickle
import os

here = os.path.dirname(__file__)

fname = os.path.join(here, '../data/nz/sobol_sequence_s8om.pkl')
with open(fname, 'rb') as f:
    data = pickle.load(f)

om = data['omega_m']
sig8= data['sigma_8']
s8 = sig8*(om/0.3)**0.5

print('omega_m range'.ljust(20)+': ({:.3f}, {:.3f})'.format(om.min(), om.max()))
print('sigma_8 range'.ljust(20)+': ({:.3f}, {:.3f})'.format(sig8.min(), sig8.max()))
print('s8 range'.ljust(20)     +': ({:.3f}, {:.3f})'.format(s8.min(), s8.max()))

# Save as txt file in cosmosis format
# adding cosmosis header describing the columns' names
d = np.array([om, s8]).T[:5, :]
header=  'cosmological_parameters--omega_m cosmological_parameters--s_8'
fname = os.path.join(here, '../data/nz/sobol_sequence_s8om.txt')
np.savetxt(fname, d, header=header)