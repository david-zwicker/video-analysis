'''
Created on Sep 18, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import subprocess

from .project import HPCProjectBase
from ..algorithm.utils import change_directory


class SlurmProject(HPCProjectBase):
    """ HPC project based on the slurm scheduler """
    
    job_files = {'pass1_single.py', 'pass1_single_slurm.sh',
                 'pass2_single.py', 'pass2_single_slurm.sh'}
        
    def submit(self):
        """ submit the tracking job using slurm """
        with change_directory(self.folder):
            # submit first job
            res = subprocess.check_output(['sbatch', 'pass1_single_slurm.sh'])
            pid_pass1 = int(res.split()[-1])
            self.pids = [pid_pass1]
            self.logger.info('Job id of first pass: %d', pid_pass1)
            
            # submit second job if requested
            if self.passes >= 2:
                res = subprocess.check_output(['sbatch',
                                               '--dependency=afterok:%d' % pid_pass1,
                                               'pass2_single_slurm.sh'])
                pid_pass2 = int(res.split()[-1])
                self.pids.append(pid_pass2)
                self.logger.info('Job id of second pass: %d', pid_pass2)
        
