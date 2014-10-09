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
    files_job = {1: ('pass1_slurm.sh', 'pass1_single.py'), 
                 2: ('pass2_slurm.sh', 'pass2_single.py'),
                 3: ('pass3_slurm.sh', 'pass3_single.py')}
    files_cleanup = {1: ('pass1_job_id.txt', 'log_pass1*'),
                     2: ('pass2_job_id.txt', 'log_pass2*'),
                     3: ('pass3_job_id.txt', 'log_pass3*')}
    

    def submit(self):
        """ submit the tracking job using slurm """
        with change_directory(self.folder):
            pid_prev = None #< pid of the previous process
            
            for pass_id in self.passes:
                # create job command
                cmd = ['sbatch']
                if pid_prev is not None:
                    cmd.append('--dependency=afterok:%d' % pid_prev)
                cmd.append(self.files_job[pass_id][0])

                # submit command and fetch pid from output
                res = sp.check_output(cmd)
                pid_prev = int(res.split()[-1])
                self.logger.info('Job id of pass %d: %d', pass_id, pid_prev)


    def check_log_for_error(self, log_file):
        """ scans a log file for errors.
        returns a string indicating the error or None """
        uri = os.path.join(self.folder, log_file)
        log = sp.check_output(['tail', '-n', '10', uri])
        if 'exceeded memory limit' in log:
            return 'exceeded-memory'
        elif 'FFmpeg encountered the following error' in log:
            return 'ffmpeg-error'
        elif 'Error' in log:
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
                # squeue does not have information yet, but the process started
                chunks = ['starting', 'nan']
            status['state'] = chunks[0].strip().lower()
            status['elapsed'] = chunks[1].strip().lower()
            
        return status


    def get_status(self):
        """ check the status of the project """
        status = {}
        if os.path.isdir(self.folder):
            # project is initialized
            status['project'] = 'initialized'

            for pass_id in xrange(1, 4):
                status['pass%d' % pass_id] = self.check_pass_status(pass_id)

        else:
            status['project'] = 'not-initialized'
        return status