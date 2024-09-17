import numpy as np
import pickle
import os

#here = os.path.dirname(__file__)

fname = '../data/nz/sobol_sequence_6dim_3Sep24_with_mnu_0_to_1000.pkl'
with open(fname, 'rb') as f:
    data = pickle.load(f)

om = data['omega_m']
s8 = data['s_8']
h0 = data['h0']
omega_b = data['omega_b']
ns = data['n_s']
mnu = data['mnu']


print('omega_m range'.ljust(20)+': ({:.3f}, {:.3f})'.format(om.min(), om.max()))
print('s8 range'.ljust(20)+': ({:.3f}, {:.3f})'.format(s8.min(), s8.max()))
print('h0 range'.ljust(20)+': ({:.3f}, {:.3f})'.format(h0.min(), h0.max()))
print('omega_b range'.ljust(20)+': ({:.3f}, {:.3f})'.format(omega_b.min(), omega_b.max()))
print('n_s range'.ljust(20)+': ({:.3f}, {:.3f})'.format(ns.min(), ns.max()))
print('mnu range'.ljust(20)+': ({:.3f}, {:.3f})'.format(mnu.min(), mnu.max()))

# Save as txt file in cosmosis format
# adding cosmosis header describing the columns' names
d = np.array([om, s8, h0, omega_b, ns, mnu]).T
header=  'cosmological_parameters--omega_m cosmological_parameters--s_8 cosmological_parameters--h0 cosmological_parameters--omega_b cosmological_parameters--n_s cosmological_parameters--mnu'
fname = '../data/nz/sobol_sequence_6dim_3Sep24_with_mnu_0_to_1000.txt'
np.savetxt(fname, d, header=header)