#!/bin/bash

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
srun cosmosis config/noisy-cosmogridc-simdv-joint-2pt-moped-run-8arcmincut-december/des-y3-shear-2pt-map3-NLA-all-008.ini
