'''
Created on Sep 18, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import subprocess as sp
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



class ProjectSingleSlurm(HPCProjectBase):
    """ HPC project based on the slurm scheduler """

    # the order of these files matters!
    job_files = ('pass1_slurm.sh', 'pass1_single.py', 
                 'pass2_slurm.sh', 'pass2_single.py')
    

    def submit(self):
        """ submit the tracking job using slurm """
        with change_directory(self.folder):
            # submit first job
            res = sp.check_output(['sbatch', self.job_files[0]])
            pid_pass1 = int(res.split()[-1])
            self.pids = [pid_pass1]
            self.logger.info('Job id of first pass: %d', pid_pass1)

            # submit second job if requested
            if self.passes >= 2:
                res = sp.check_output(['sbatch',
                                       '--dependency=afterok:%d' % pid_pass1,
                                       self.job_files[2]])
                pid_pass2 = int(res.split()[-1])
                self.pids.append(pid_pass2)
                self.logger.info('Job id of second pass: %d', pid_pass2)


    def check_log_for_error(self, log_file):
        """ scans a log file for errors.
        returns a string indicating the error or None """
        uri = os.path.join(self.folder, log_file)
        log = sp.check_output(['tail', '-n', '10', uri])
        for line in log.splitlines():
            if 'exceeded memory limit' in line:
                return 'exceeded-memory'
            elif 'FFmpeg encountered the following error' in line:
                return 'ffmpeg-error'
            elif 'Error' in line:
                return 'error'
            
        return None


    def check_pass_status(self, pass_id):
        """ check the status of a single pass """
        status = {}

        # check whether slurm job has been initialized
        pid_file = os.path.join(self.folder, 'pass%d_job_id.txt' % pass_id)
        try:
            pids = open(pid_file).readlines()
        except IOError:
            status['state'] = 'not-started'
            return status
        else:
            status['state'] = 'started'

        # check the status of the job
        pid = pids[-1].strip()
        status['job-id'] = int(pid)
        try:
            res = sp.check_output(['squeue', '-j', pid,
                                   '-o', '%T|%M'], #< output format
                                  stderr=sp.STDOUT)
            
        except sp.CalledProcessError as err:
            if 'Invalid job id specified' in err.output:
                # job seems to have finished already
                res = sp.check_output(['sacct', '-j', pid, '-P',
                                       '-o', 'state,MaxRSS,Elapsed,cputime'])
                
                try:
                    chunks = res.splitlines()[1].split('|')
                except IndexError:
                    self.logger.warn(res)
                    chunks = ['unknown', 'nan', 'nan', 'nan']
                status['state'] = chunks[0].strip().lower()
                status['elapsed'] = chunks[2].strip()

                # check output for error
                log_file = 'log_pass%d_%s.txt' % (pass_id, pid)
                log_error = self.check_log_for_error(log_file)
                if log_error is not None:
                    status['state'] = log_error

            else:
                # unknown error
                self.logger.warn(err.output)

        else:
            # jobs seems to be currently running
            try:
                chunks = res.splitlines()[1].split('|')
            except IndexError:
                self.logger.warn(res)
                chunks = ['unknown', 'nan']
            status['state'] = chunks[0].strip().lower()
            status['elapsed'] = chunks[1].strip().lower()
            
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