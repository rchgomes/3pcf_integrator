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
srun cosmosis config/emulator-train-tatt/december-2025/generate_training_set_ddE_3600to4500.ini
