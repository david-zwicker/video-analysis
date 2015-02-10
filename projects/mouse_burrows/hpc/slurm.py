'''
Created on Sep 18, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import subprocess as sp
import os

import yaml

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

    # The first file of each pass will be the one submitted to slurm
    files_job = {1: ('pass1_slurm.sh', 'pass1_single.py'), 
                 2: ('pass2_slurm.sh', 'pass2_single.py'),
                 3: ('pass3_slurm.sh', 'pass3_single.py'),
                 4: ('pass4_slurm.sh', 'pass4_single.py')}
    files_cleanup = {1: ('pass1_job_id.txt', 'status_pass1.yaml', 'log_pass1*'),
                     2: ('pass2_job_id.txt', 'status_pass2.yaml', 'log_pass2*'),
                     3: ('pass3_job_id.txt', 'status_pass3.yaml', 'log_pass3*'),
                     4: ('pass4_job_id.txt', 'status_pass4.yaml', 'log_pass4*')}
    
    # file name patterns used here
    job_id_file = 'pass%d_job_id.txt'
    job_ids_file = 'job_ids.txt'
    log_file = 'log_pass%d_%s.txt'
    status_cache_file = 'status_pass%d.yaml'
    status_file = '%s_status.yaml'
    pass_finished_states = {'done', 'ffmpeg-error'}


    def prepare_project(self, *args, **kwargs):
        """ prepare the work directory by setting up all necessary files """
        # read job_ids from the designated file
        job_id_file = os.path.join(self.folder, self.job_ids_file)
        job_ids = {}
        try:
            for line in open(job_id_file):
                # extract numeric pass id and job id
                p_id, j_id = line.split('-')
                p_id = int(p_id.split(' ')[1])
                if p_id in job_ids:
                    # this job has been restarted => all previous jobs should
                    # be finished by now
                    job_ids = {'p_id': int(j_id)}
                else:
                    # add this id to list of jobs that may still be running
                    job_ids[p_id] = int(j_id)
        except IOError:
            pass
                
        # cancel the jobs that were found
        if job_ids:
            cmd = ['scancel', '-Q'] + [str(i) for i in job_ids.values()]
            sp.check_call(cmd)
        
        # add files using 
        super(ProjectSingleSlurm, self).prepare_project(*args, **kwargs)


    def submit(self):
        """ submit the tracking job using slurm """
        with change_directory(self.folder):
            job_id = None #< job_id of the previous process
            
            # submit new jobs
            for pass_id in self.passes:
                # create job command
                cmd = ['sbatch']
                if job_id is not None:
                    cmd.append('--dependency=afterok:%d' % job_id)
                cmd.append(self.files_job[pass_id][0])

                # submit command and fetch job_id from output
                res = sp.check_output(cmd)
                job_id = int(res.split()[-1])
                # save job_id to file
                with open(self.job_ids_file, 'a') as f:
                    f.write('pass %d - %d\n' % (pass_id, job_id))
                self.logger.info('Job id of pass %d: %d', pass_id, job_id)


    def check_log_for_error(self, log_file):
        """ scans a log file for errors.
        returns a string indicating the error or None """
        uri = os.path.join(self.folder, log_file)
        try:
            log = sp.check_output(['tail', '-n', '10', uri])
        except sp.CalledProcessError:
            return '???'
        if 'exceeded memory limit' in log:
            return 'memory-error'
        elif 'FFmpeg encountered the following error' in log:
            return 'ffmpeg-error'
        elif 'Error' in log:
            return 'error'
            
        return None


    def check_pass_status(self, pass_id):
        """ check the status of a single pass by using slurm commands """
        status = {}

        # check whether slurm job has been initialized
        pid_file = os.path.join(self.folder, self.job_id_file % pass_id)
        try:
            pids = open(pid_file).readlines()
        except IOError:
            status['state'] = 'pending'
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
                state = chunks[0].strip().lower()
                status['state'] = state.replace('completed', 'done')
                status['elapsed'] = chunks[2].strip()

                # check output for error
                log_file = self.log_file % (pass_id, pid)
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
                chunks = ['unknown', 'nan']
            status['state'] = chunks[0].strip().lower()
            status['elapsed'] = chunks[1].strip().lower()
            
        return status


    def get_pass_status(self, pass_id):
        """ check the status of a single pass
        This function is deprecated, because we now store the status information
        in a single yaml file per project
        """
        cache_file_name = os.path.join(self.folder,
                                       self.status_cache_file % pass_id)
        try:
            # try loading the status from the cache file
            with open(cache_file_name, 'r') as infile:
                status = yaml.load(infile)
                
        except IOError:
            # determine the status by querying slurm
            status = self.check_pass_status(pass_id)
            
            # save the status to the cache file if the pass has finished
            if status['state'] in self.pass_finished_states:
                with open(cache_file_name, 'w') as outfile:
                    yaml.dump(status, outfile)
            
        return status
        

    def get_status(self):
        """ check the status of the project """
        status = {}
        if os.path.isdir(self.folder):
            # project is initialized
            status['project'] = 'initialized'

            # load yaml file to get status of project
            status_path = os.path.join(self.folder,
                                       self.status_file % self.name)
            try:
                with open(status_path, "r") as fp:
                    status_data = yaml.load(fp)
            except IOError:
                status_data = {}

            # get the status of the individual passes
            status_updated = False
            for pass_id in xrange(1, 5):
                pass_name = 'pass%d' % pass_id
                if pass_name in status_data:
                    status[pass_name] = status_data[pass_name]
                else:
                    status[pass_name] = {'state': 'unknown'}
                    
                if status[pass_name]['state'] not in {'done', 'error'}:   
                    status[pass_name] = self.check_pass_status(pass_id)
                    status_data[pass_name] = status[pass_name]
                    status_updated = True
                    
            # write updated status
            if status_updated:
                with open(status_path, "w") as fp:
                    yaml.dump(status_data, fp)

        else:
            status['project'] = 'not-initialized'
        return status
    
    