import sys
import os
import cosmosis

cosmosis_dir = os.path.dirname(cosmosis.__file__)
sampler_dir = os.path.join(cosmosis_dir, 'samplers')
dest = os.path.join(sampler_dir, 'moped')

here = os.path.dirname(__file__)
source = os.path.join(here, '../python/moped')

# create folder
if not os.path.exists(dest):
    print('creating folder at {}'.format(dest))
    os.makedirs(dest)

# copy contents
command = 'cp {}/* {}'.format(source, dest)
print(command)
os.system(command)

# register
init_file = os.path.join(sampler_dir, '__init__.py')
import_string = 'from .moped.moped_sampler import MOPEDSampler'
exists = False
with open(init_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if import_string in line:
            exists = True
if not exists:
    with open(init_file, 'a') as f:
        f.write(import_string+'\n')