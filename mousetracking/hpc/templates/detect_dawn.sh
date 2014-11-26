#!/bin/bash

#SBATCH -n 1                 # Number of cores
#SBATCH -t 10*60*60          # Runtime in minutes
#SBATCH -p general           # Partition to submit to
#SBATCH --mem-per-cpu=500    # Memory per cpu in MB (see also --mem)
#SBATCH -o {JOB_DIRECTORY}/log_detect_dawn_%j.txt    # File to which stdout and stderr will be written
#SBATCH --job-name=DD_%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=dzwicker@seas.harvard.edu

echo "Start job with id $SLURM_JOB_ID"

# load python environment
source ~/.profile
# change to job directory
cd {JOB_DIRECTORY}
# run python script
{SCRIPT_DIRECTORY}/detect_dawn.py "{VIDEO_FILE}"

echo "Ended job with id $SLURM_JOB_ID"
