#!/bin/bash

#SBATCH -n {PASS1_CORES}     # Number of cores
#SBATCH -N 1                 # Ensure that all cores are on one machine
#SBATCH -t {PASS1_TIME}      # Runtime in minutes
#SBATCH -p {PARTITION}       # Partition to submit to
#SBATCH ----mem-per-cpu={PASS1_MEMORY} # Memory per cpu in MB (see also --mem)
#SBATCH -o {JOB_DIRECTORY}/log_pass1_%j.txt    # File to which stdout and stderr will be written
#SBATCH --mail-type=FAIL
#SBATCH --mail-user={USER_EMAIL}

echo "Start job with id $SLURM_JOB_ID"

# load python environment
source ~/.profile
# change to job directory
cd {JOB_DIRECTORY}
# run python script
python pass1_single.py

echo "Ended job with id $SLURM_JOB_ID"
