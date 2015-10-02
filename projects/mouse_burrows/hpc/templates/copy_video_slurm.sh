#!/bin/bash

#SBATCH -n 1                 # Number of cores
#SBATCH -t 24:00             # Runtime in minutes
#SBATCH -p serial_requeue    # Partition to submit to
#SBATCH --mem-per-cpu=500    # Memory per cpu in MB (see also --mem)
#SBATCH -o {JOB_DIRECTORY}/log_copy_video_%j.txt    # File to which stdout and stderr will be written
#SBATCH --job-name=C_{NAME}
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=dzwicker@seas.harvard.edu

echo "Start job with id $SLURM_JOB_ID"

# copy video to temporary location if necessary
rsync -avzh --progress "{VIDEO_FILE_SOURCE}" "{VIDEO_FILE_TEMPORARY}"

echo "Ended job with id $SLURM_JOB_ID"
