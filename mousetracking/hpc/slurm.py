'''
Created on Sep 18, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import subprocess
import os

from .project import HPCProjectBase
from ..algorithm.utils import change_directory



def parse_time(text):
    """ parse time of slurm output """
    # check for days
    tokens = text.split('-', 1)
    if len(tokens) > 1:
        days = int(tokens[0])
        rest = tokens[1]
    else:
        days = 0
        rest = tokens[0]

    # check for time
    tokens = rest.split(':')
    return days*24*60 + int(tokens[0])*60 + int(tokens[1])



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
        
        
    def check_pass_status(self, pass_id):
        """ check the status of a single pass """
        status = {}
        
        # check whether slurm job has been initialized
        pid_file = os.path.join(self.folder, 'pass%d_job_id.txt' % pass_id)
        try:
            pids = open(pid_file).readlines()
        except OSError:
            status['general'] = 'not-started'
            return status
        else:
            status['general'] = 'started'

        # check the status of the job
        pid = int(pids[-1])
        res = subprocess.check_output(['squeue', '-j', pid,
                                       '-o', '"%T|%M"']) #< output format
        if 'Invalid job id specified' in res:
            # job seems to have finished already
            res = subprocess.check_output(['sacct', '-j', pid, '-P',
                                           '-o', 'state,MaxRSS,Elapsed,cputime'])
            chunks = res.split('|')
            status['state'] = chunks[0]
            status['elapsed'] = chunks[2]
            
        else:
            # jobs seems to be currently running
            chunks = res.split('|')
            status['state'] = chunks[0]
            status['elapsed'] = chunks[1]
            
        return status
        

    def get_status(self):
        """ check the status of the project """
        status = {}
        if os.path.isdir(self.folder):
            # project is initialized
            status['project'] = 'initialized'
            
            status['pass1'] = self.check_pass_status(1)
            status['pass2'] = self.check_pass_status(2)
            
        else:
            status['project'] = 'not-initialized'
        return status
            