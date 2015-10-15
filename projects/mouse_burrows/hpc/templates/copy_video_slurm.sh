#!/bin/bash

#SBATCH -n {PASS0/CORES}     # Number of cores
#SBATCH -t {PASS0/TIME}      # Runtime in minutes
#SBATCH -p serial_requeue    # Partition to submit to
#SBATCH --mem-per-cpu={PASS0/MEMORY}   # Memory per cpu in MB (see also --mem)
#SBATCH -o {JOB_DIRECTORY}/log_copy_video_%j.txt    # File to which stdout and stderr will be written
#SBATCH --job-name=C_{NAME}
#SBATCH --mail-type=FAIL
#SBATCH --mail-user={NOTIFICATION_EMAIL}

echo "Start job with id $SLURM_JOB_ID"

# copy video to temporary location if necessary
mkdir -p {VIDEO_FOLDER_TEMPORARY}
rsync -avzh --progress {VIDEO_FILE_SOURCE} {VIDEO_FOLDER_TEMPORARY}

echo "Ended job with id $SLURM_JOB_ID"
