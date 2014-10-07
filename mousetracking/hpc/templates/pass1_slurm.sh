#!/bin/bash

#SBATCH -n {PASS1/CORES}     # Number of cores
#SBATCH -N 1                 # Ensure that all cores are on one machine
#SBATCH -t {PASS1/TIME}      # Runtime in minutes
#SBATCH -p {SLURM_PARTITION} # Partition to submit to
#SBATCH --mem-per-cpu={PASS1/MEMORY} # Memory per cpu in MB (see also --mem)
#SBATCH -o {JOB_DIRECTORY}/log_pass1_%j.txt    # File to which stdout and stderr will be written
#SBATCH --job-name=P1_{NAME}
#SBATCH --mail-type=FAIL
#SBATCH --mail-user={NOTIFICATION_EMAIL}

echo "Start job with id $SLURM_JOB_ID"
echo $SLURM_JOB_ID >> pass1_job_id.txt

# load python environment
source ~/.profile
# change to job directory
cd {JOB_DIRECTORY}
# run python script
python {JOB_FILE_1}

echo "Ended job with id $SLURM_JOB_ID"
