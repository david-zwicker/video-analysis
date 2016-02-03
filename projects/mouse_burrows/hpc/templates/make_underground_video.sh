#!/bin/bash

#SBATCH -n 1                 # Number of cores
#SBATCH -t {PASS9/TIME}      # Runtime in minutes
#SBATCH -p {SLURM_PARTITION} # Partition to submit to
#SBATCH --mem-per-cpu={PASS9/MEMORY}   # Memory per cpu in MB (see also --mem)
#SBATCH -o {JOB_DIRECTORY}/log_underground_video_%j.txt    # File to which stdout and stderr will be written
#SBATCH --job-name=U_{NAME}
#SBATCH --mail-type=FAIL
#SBATCH --mail-user={NOTIFICATION_EMAIL}

echo "Start job with id $SLURM_JOB_ID"

# load python environment
source ~/.profile
# change to job directory
cd {JOB_DIRECTORY}
# run script to create underground movie
~/Code/video-analysis/projects/mouse_burrows/scripts/get_underground_movie.py \
    --result_file {JOB_DIRECTORY}/{NAME}_results.yaml \
    --scale_bar

echo "Ended job with id $SLURM_JOB_ID"
