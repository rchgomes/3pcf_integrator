import os

# Create the directory
dir_name = "submit_noisy_validation/"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# Base content for the .ini files (without the last line)
base_content = """#!/bin/bash

#SBATCH -A des 
#SBATCH -C cpu 
#SBATCH -q regular 
#SBATCH -t 24:00:00 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --ntasks=1

cd /global/cfs/cdirs/des/rchgoms1/3pcf_integrator
module load python
conda activate csis3pcf

source setup-3pcf-path.sh
"""

# Generate 50 files with appropriate content
for i in range(50):
    file_name = f"2pt_map3_{i}.ini"
    file_path = os.path.join(dir_name, file_name)

    # Last line specific to each file
    last_line = f"srun cosmosis config/noisy-dv-validation/cosmogrid-shear-2pt-map3-NLA-all-noisy-dv-{i}.ini\n"

    # Write the file
    with open(file_path, 'w') as f:
        f.write(base_content)
        f.write(last_line)

print("Files created successfully!")
