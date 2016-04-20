'''
Created on Jul 31, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

This package provides class definitions for referencing a single video file.

This code has been modified from the project moviepy, which is released under
the MIT license at github:
https://github.com/Zulko/moviepy/blob/master/moviepy/video/io

The MIT license text is included in the present package in the file
/external/LICENSE_MIT.txt
'''

from __future__ import division

import fcntl
import os
import re
import logging
import subprocess
import time

import numpy as np

from .base import VideoBase
from utils import cache

logger = logging.getLogger('video.io')

# find the file handle to /dev/null to dumb strings
try:
    from subprocess import DEVNULL # py3k
except ImportError:
    DEVNULL = open(os.devnull, 'wb')


def get_ffmpeg_version(cmd):
    """ helper function trying to get the version number from ffmpeg """    
    try:
        # try getting help page from ffmpeg
        output = subprocess.check_output([cmd, '-h'], stderr=subprocess.STDOUT)
        # search for the version number and parse it
        match = re.search("ffmpeg version (\d+)\.(\d+)", output)
        version = tuple(int(match.group(k)) for k in xrange(1, 3))
    except:
        version = None
    return version


# search for the FFmpeg command
FFMPEG_VERSION = get_ffmpeg_version('ffmpeg')
if FFMPEG_VERSION:
    FFMPEG_BINARY = 'ffmpeg'
    FFPROBE_BINARY = 'ffprobe'
    logger.debug('Found ffmpeg v%s at %s',
                 '.'.join(str(i) for i in FFMPEG_VERSION), 
                 subprocess.check_output(['which', 'ffmpeg']).strip())
else:
    FFMPEG_VERSION = get_ffmpeg_version('ffmpeg.exe')
    if FFMPEG_VERSION:
        FFMPEG_BINARY = 'ffmpeg.exe'
        FFPROBE_BINARY = 'ffprobe.exe'
        logger.debug('Found ffmpeg.exe v%s.',
                     '.'.join(str(i) for i in FFMPEG_VERSION))
    else:
        FFMPEG_BINARY = None
        FFPROBE_BINARY = None
        logger.warn('ffmpeg binary not found. Functions relying on it will not '
                    'be available.')



class FFmpegError(IOError):
    pass



class VideoFFmpeg(VideoBase):
    """ Class handling a single movie file using FFmpeg
    """ 
    
    #seek_max_frames = 100 
    seekable = True #< this video is seekable
    parameters_default = {
        'bufsize': None,    #< buffer size for communicating with ffmpeg
        'pix_fmt': 'rgb24', #< pixel format returned by ffmpeg
        'video_info_method': 'header', #< method for estimating frame count
        'ffprobe_cache': None, #< cache file for the ffprobe result
        'reopen_delay': 0, #< seconds to wait before reopening a video
        'seek_method': 'auto', #< method used for seeking
        'seek_max_frames': 100, #< the maximal number of frames we seek through
        'seek_offset': 1, #< seconds the rough seek is placed before the target
    }
    

    def __init__(self, filename, parameters=None):
        """ initialize a video that will be read with FFmpeg
        filename is the name of the filename.
        `parameters` denotes additional parameters 
        """
        self.parameters = self.parameters_default.copy()
        if parameters:
            self.parameters.update(parameters)

        # get information about the video using FFmpeg
        self.filename = os.path.expanduser(filename)
        
        if self.parameters['video_info_method'] == 'header':
            # use the information from the movie header
            infos = ffmpeg_parse_infos(self.filename)
        elif self.parameters['video_info_method'] == 'ffprobe':
            # determine the information by iterating through the video
            infos = ffprobe_get_infos(
                        self.filename,
                        cache_file=self.parameters['ffprobe_cache']
                    )
        else:
            raise ValueError('Unknown method `%s` for determining information '
                             'about the video'
                             % self.parameters['video_info_method'])
        
        # store information in class
        self.duration = infos['video_duration']
        self.infos = infos

        self.pix_fmt = self.parameters['pix_fmt']
        if self.pix_fmt == 'rgba':
            self.depth = 4
        elif self.pix_fmt == 'rgb24':
            self.depth = 3
        else:
            raise ValueError('Unsupported pixel format `%s`' % self.pix_fmt)

        if self.parameters['bufsize'] is None:
            w, h = infos['video_size']
            bufsize = 2 * self.depth * w * h + 100 #< add some safety margin

        # initialize the process that eventually reads the video
        self.bufsize = bufsize
        self.proc = None
        self.open()

        self.lastread = None
        
        super(VideoFFmpeg, self).__init__(size=tuple(infos['video_size']),
                                          frame_count=infos['video_nframes'],
                                          fps=infos['video_fps'], is_color=True)

        logger.debug('Initialized video `%s` with %d frames using FFmpeg',
                     self.filename, infos['video_nframes'])


    def print_infos(self):
        """ print information about the video file """
        if self.parameters['video_info_method'] == 'header':
            ffmpeg_parse_infos(self.filename, print_infos=True)
        elif self.parameters['video_info_method'] == 'ffprobe':
            print(self.infos)
        else:
            raise ValueError('Unknown method `%s` for determining information '
                             'about the video'
                             % self.parameters['video_info_method'])


    @property
    def closed(self):
        return self.proc is None


    def open(self, index=0):
        """ Opens the file, creates the pipe. """
        logger.debug('Open video `%s`' % self.filename)
        
        # close video if it was opened
        if not self.closed:
            self.close() 
        
            # wait some time until we reopen the video 
            reopen_delay = self.parameters['reopen_delay']
            if reopen_delay > 0:
                logger.debug('Wait %g seconds before reopening video', 
                             reopen_delay)
                time.sleep(reopen_delay)

        if index > 0:
            # we have to seek to another index/time

            # determine the time that we have to seek to
            # the -0.1 is necessary to prevent rounding errors
            starttime = (index - 0.1) / self.fps
            
            # determine which method to use for seeking
            seek_method = self.parameters['seek_method']
            if seek_method == 'auto':
                if FFMPEG_VERSION > (2, 1):
                    seek_method = 'exact'
                else:
                    seek_method = 'keyframe' 
            
            if seek_method == 'exact':
                # newer ffmpeg version, which supports accurate seeking
                i_arg = ['-ss', "%.03f" % starttime,
                         '-i', self.filename]
                
            elif seek_method == 'keyframe':
                # older ffmpeg version, which does not support accurate seeking
                if index < self.parameters['seek_max_frames']:
                    # we only have to seek a little bit
                    i_arg = ['-i', self.filename,
                             '-ss', "%.03f" % starttime]
                else:
                    # we first seek to a keyframe and then proceed from there
                    seek_offset = self.parameters['seek_offset']
                    i_arg = ['-ss', "%.03f" % (starttime - seek_offset),
                             '-i', self.filename,
                             '-ss', "%.03f" % seek_offset]
                    
            else:
                raise ValueError('Unknown seek method `%s`' % seek_method)

            logger.debug('Seek video to frame %d (=%.03fs)', index, starttime)
            
        else:
            # we can just open the video at the first frame
            i_arg = ['-i', self.filename]
        
        # build ffmpeg command line
        cmd = ([FFMPEG_BINARY] + 
               i_arg +
               ['-loglevel', 'error', 
                '-f', 'image2pipe',
                '-pix_fmt', self.pix_fmt,
                '-vcodec', 'rawvideo',
                '-'])
        
        self.proc = subprocess.Popen(cmd, bufsize=self.bufsize,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
        
        # set the stderr to non-blocking; used the idea from
        #     http://stackoverflow.com/a/8980466/932593
        # this only works on UNIX!
        fcntl.fcntl(self.proc.stderr.fileno(), fcntl.F_SETFL, os.O_NONBLOCK)
        
        self._frame_pos = index


    def set_frame_pos(self, index):
        """ sets the video to position index """
        if index != self._frame_pos:
            # determine the farthest frame that we would reach by skipping
            max_future_frame = self._frame_pos + \
                               self.parameters['seek_max_frames']
                               
            if (index < self._frame_pos) or (index > max_future_frame):
                # reopen the video at the correct position
                self.open(index)
            else:
                # skip frames to reach the requested position
                skip_frames = index - self._frame_pos
                w, h = self.size
                for _ in xrange(skip_frames):
                    self.proc.stdout.read(self.depth*w*h)
                    self.proc.stdout.flush()
                self._frame_pos = index
                

    def get_next_frame(self):
        """ retrieve the next frame from the video """
        # read standard error output and log it if requested
        try:
            stderr_content = self.proc.stderr.read()
        except IOError:
            # nothing to read from stderr
            pass
        else:
            logger.debug(stderr_content)
        
        w, h = self.size
        nbytes = self.depth*w*h

        # read the next frame from the process 
        s = self.proc.stdout.read(nbytes)

        if len(s) != nbytes:
            # frame_count is a rather crude estimate of the length of the video.
            # We thus stop the iteration, when we think we are close to the end.
            # The magic numbers 5 and 0.01 are rather arbitrary, but have proven
            # to work in most practical cases.
            frames_remaining = self.frame_count - self._frame_pos 
            if frames_remaining < 5 or frames_remaining < 0.01*self.frame_count:
                raise StopIteration
            
            logger.warn("Warning: in file %s, %d bytes wanted but %d bytes "
                        "read, at frame %d/%d, at time %.02f/%.02f sec. "
                        "Using the last valid frame instead." %
                        (self.filename, nbytes, len(s),
                         self._frame_pos, self.frame_count,
                         self._frame_pos/self.fps, self.duration))
            
            if self.lastread is None:
                raise FFmpegError(
                    "Failed to read the first frame of video file %s. That "
                    "might mean that the file is corrupted. That may also mean "
                    "that your version of FFmpeg (%s) is too old."
                    % (self.filename, '.'.join(str(v) for v in FFMPEG_VERSION))
                )

            result = self.lastread

        else:
            # frame has been obtained properly
            shape = (h, w, self.depth)
            result = np.frombuffer(s, dtype='uint8').reshape(shape)
            self.lastread = result
            self._frame_pos += 1
            
        return result


    def get_frame(self, index):
        """ Read a file video frame at time t.
        
        Note for coders: getting an arbitrary frame in the video with
        FFmpeg can be painfully slow if some decoding has to be done.
        This function tries to avoid fetching arbitrary frames
        whenever possible, by moving between adjacent frames.
        """
        if index == self._frame_pos - 1:
            return self.lastread
        else:
            self.set_frame_pos(index)
            result = self.get_next_frame()
            assert self._frame_pos == index + 1
            return result

    
    def close(self):
        """ close the process reading the video """
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.stdout.close()
                self.proc.stderr.close()
                self.proc.wait()
            except IOError:
                pass
            self.proc = None

    
    def __enter__(self):
        return self
    
        
    def __exit__(self, e_type, e_value, e_traceback):
        self.close()

    
    def __del__(self):
        self.close()
        if hasattr(self, 'lastread'):
            del self.lastread
            


class VideoWriterFFmpeg(object):
    """ A class for FFmpeg-based video writing.
    
    A class to write videos using FFmpeg. FFmpeg will write in a large
    choice of formats.
    
    Parameters
    -----------
    
    filename
      Any filename like 'video.mp4' etc. but if you want to avoid
      complications it is recommended to use the generic extension
      '.avi' for all your videos.
    
    size
      Size (width, height) of the output video in pixels.
      
    fps
      Frames per second in the output video file.
      
    codec
      FFmpeg codec. It seems that in terms of quality the hierarchy is
      'rawvideo' = 'png' > 'mpeg4' > 'libx264'
      'png' manages the same lossless quality as 'rawvideo' but yields
      smaller files. Type ``FFmpeg -codecs`` in a terminal to get a list
      of accepted codecs.

      Note for default 'libx264': by default the pixel format yuv420p
      is used. If the video dimensions are not both even (e.g. 720x405)
      another pixel format is used, and this can cause problem in some
      video readers.

    bitrate
      Only relevant for codecs which accept a bitrate. "5000k" offers
      nice results in general.
    
    """
        
    def __init__(self, filename, size, fps, is_color=True, codec="libx264",
                 bitrate=None):
        """
        Initializes the video writer.
        `filename` is the name of the video
        `size` is a tuple determining the width and height of the video
        `fps` determines the frame rate in 1/seconds
        `is_color` is a flag indicating whether the video is in color
        `codec` selects a codec supported by FFmpeg
        `bitrate` determines the associated bitrate
        """
        
        self.filename = os.path.expanduser(filename)
        self.codec = codec
        self.ext = self.filename.split(".")[-1]
        self.size = size
        self.is_color = is_color
        self.frames_written = 0   

        if size[0] % 2 != 0 or size[1] % 2 != 0:
            raise ValueError('Both dimensions of the video must be even for '
                             'the video codec to work properly')

        # determine whether we are in debug mode
        debug = (logger.getEffectiveLevel() >= logging.DEBUG)

        #FIXME: consider adding the flags
        # "-f ismv"  "-movflags frag_keyframe"
        # to avoid corrupted mov files, if movie writing is interrupted
        
        # build the FFmpeg command
        cmd = (
            [FFMPEG_BINARY, '-y',
             '-loglevel', 'verbose' if debug else 'error',
             '-threads', '1', #< single threaded encoding for safety
             '-f', 'rawvideo',
             '-vcodec','rawvideo',
             '-s', "%dx%d" % tuple(size),
             '-pix_fmt', 'rgb24' if is_color else 'gray',
             '-r', '%.02f' % fps,
             '-i', '-',
             '-an'] # no audio
            + ([] if (codec is None) else ['-c:v', codec])
            + ([] if (bitrate is None) else ['-b:v', bitrate])

            # http://trac.FFmpeg.org/ticket/658
            + (['-pix_fmt', 'yuv420p']
               if ((codec == 'libx264') and
                   (size[0] % 2 == 0) and
                   (size[1] % 2 == 0))
               else [])

            + ['-r', "%.02f" % fps, filename]
        )
        
        # estimate the buffer size with some safety margins
        depth = 3 if is_color else 1
        bufsize = 2 * depth * size[0] * size[1] + 100
        
        # start FFmpeg, which should wait for input
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=DEVNULL, stderr=subprocess.PIPE,
                                     bufsize=bufsize)
        
        # set the stderr to non-blocking; used the idea from
        #     http://stackoverflow.com/a/8980466/932593
        # this only works on UNIX!
        fcntl.fcntl(self.proc.stderr.fileno(), fcntl.F_SETFL, os.O_NONBLOCK)

        logger.info('Start writing video `%s` with codec `%s` using FFmpeg.',
                    filename, codec)

    
    @property
    def shape(self):
        """ returns the shape of the data describing the movie """
        shape = (self.size[1], self.size[0])
        if self.is_color:
            shape += (3,)
        return shape
    
        
    def write_frame(self, img_array):
        """ Writes a single frame in the file """
        
        img_array = img_array.astype(np.uint8, copy=False)   
        
        if self.is_color and img_array.ndim == 2:            
            img_array = img_array[:, :, None]*np.ones((1, 1, 3), np.uint8)
            
        try:
            self.proc.stdin.write(img_array.tostring())
        except IOError as err:
            FFmpeg_error = self.proc.stderr.read()
            
            error = (str(err) +
                     "\n\nFFmpeg encountered the following error while "
                     "writing file %s:\n\n" % self.filename +
                     FFmpeg_error)

            if "Unknown encoder" in FFmpeg_error:
                error = error + ("\n\nThe video export "
                  "failed because FFmpeg didn't find the specified "
                  "codec for video encoding (%s). Please install "
                  "this codec or use a different codec") % (self.codec)
            
            elif "incorrect codec parameters ?" in FFmpeg_error:
                error = error + ("\n\nThe video export "
                  "failed, possibly because the codec specified for "
                  "the video (%s) is not compatible with the given "
                  "extension (%s). Please specify a valid 'codec' "
                  "argument in write_videofile. This would be 'libx264' "
                  "or 'mpeg4' for mp4, 'libtheora' for ogv, 'libvpx' "
                  "for webm.") % (self.codec, self.ext)

            elif  "encoder setup failed" in FFmpeg_error:
                error = error + ("\n\nThe video export "
                  "failed, possibly because the bitrate you specified "
                  "was too high or too low for the video codec.")
            
            # add parameters of the video for additional information
            error += "\nVideo: {size} {color}\nCodec: {codec}\n".format(
                        **{'size': ' x '.join(str(v) for v in self.size),
                           'color': 'color' if self.is_color else 'monochrome',
                           'codec': self.codec})
            
            raise FFmpegError(error)

        else:
            self.frames_written += 1   
        
        # read standard error output and log it if requested
        try:
            stderr_content = self.proc.stderr.read()
        except IOError:
            # nothing to read from stderr
            pass
        else:
            logger.debug(stderr_content)
        
        
    def close(self):
        """ finishes the process, which should also make the video available """
        if self.proc is not None:
            try:
                self.proc.communicate()
            except IOError:
                pass
            logger.info('Wrote video to file `%s`', self.filename)
            self.proc = None
    
    
    def __enter__(self):
        return self
    
        
    def __exit__(self, e_type, e_value, e_traceback):
        self.close()

    
    def __del__(self):
        self.close()



def ffmpeg_parse_infos(filename, print_infos=False):
    """Get file information using FFmpeg.

    Returns a dictionary with the fields:
    "video_found", "video_fps", "duration", "video_nframes",
    "video_duration"

    "video_duration" is slightly smaller than "duration" to avoid
    fetching the uncompleted frames at the end, which raises an error.
    """

    # open the file in a pipe, provoke an error, read output
    is_GIF = filename.endswith('.gif')
    cmd = [FFMPEG_BINARY, "-i", filename]
    if is_GIF:
        cmd += ["-f", "null", "/dev/null"]
    proc = subprocess.Popen(cmd,
                            bufsize=10**5,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

    proc.stdout.readline()
    proc.terminate()
    infos = proc.stderr.read().decode('utf8')
    del proc

    if print_infos:
        # print the whole info text returned by FFmpeg
        print(infos)

    lines = infos.splitlines()
    if "No such file or directory" in lines[-1]:
        raise IOError("The file %s could not be found!\n"
                      "Please check that you entered the correct path.\n"
                      "Here are the file information returned by FFmpeg:\n\n%s"
                      % (filename, infos))
    
    # get duration (in seconds)
    result = {}
    try:
        keyword = ('frame=' if is_GIF else 'Duration: ')
        line = [l for l in lines if keyword in l][0]
        match = re.findall("([0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9])",
                           line)[0]
        result['duration'] = time_to_seconds(match)
    except:
        raise IOError("Failed to read the duration of file %s.\n"
                      "Here are the file information returned by FFmpeg:\n\n%s"
                      % (filename, infos))

    # get the output line that speaks about video
    lines_video = [l for l in lines if ' Video: ' in l]
    
    result['video_found'] = bool(lines_video)
    
    if result['video_found']:
        
        line = lines_video[0]

        # get the size, of the form 460x320 (w x h)
        match = re.search(" [0-9]*x[0-9]*(,| )", line)
        s = list(map(int, line[match.start():match.end()-1].split('x')))
        result['video_size'] = s

        # get the frame rate. Sometimes it's 'tbr', sometimes 'fps', sometimes
        # tbc, and sometimes tbc/2...
        # Current policy: Trust tbr first, then fps. If result is near from 
        # x*1000/1001 where x is 23,24,25,50, replace by x*1000/1001 (very 
        # common case for the fps).
        
        try:
            match = re.search("( [0-9]*.| )[0-9]* tbr", line)
            tbr = float(line[match.start():match.end()].split(' ')[1])
            result['video_fps'] = tbr

        except:
            match = re.search("( [0-9]*.| )[0-9]* fps", line)
            substr = line[match.start():match.end()]
            result['video_fps'] = float(substr.split(' ')[1])

        # It is known that a fps of 24 is often written as 24000/1001
        # but then FFmpeg nicely rounds it to 23.98, which we hate.
        coef = 1000.0/1001.0
        fps = result['video_fps']
        for x in [23, 24, 25, 30, 50, 60]:
            if (fps != x) and abs(fps - x*coef) < .01:
                result['video_fps'] = x*coef

        result['video_nframes'] = int(result['duration']*result['video_fps']) + 1

        result['video_duration'] = result['duration']
        # We could have also recomputed the duration from the number
        # of frames, as follows:
        # >>> result['video_duration'] = result['video_nframes'] / result['video_fps']

    return result



def ffprobe_get_infos(video_file, print_infos=False, cache_file=None):
    """Get file information using ffprobe, which iterates through the video.

    Returns a dictionary with the fields:
    "video_found", "video_fps", "duration", "video_nframes",
    "video_duration"

    "video_duration" is slightly smaller than "duration" to avoid
    fetching the uncompleted frames at the end, which raises an error.
    """
    import json
    
    # prepare program call
    cmd = [FFPROBE_BINARY,
           '-i', video_file,
           '-print_format', 'json',
           '-loglevel', 'error',
           '-show_streams', '-count_frames',
           '-select_streams', 'v']
    
    if cache_file:
        # load the cache of all the ffprobe calls
        ffprobe_cache = cache.PersistentDict(cache_file)
        try:
            # try to fetch the output from this cache
            output = ffprobe_cache[video_file]
            
        except KeyError:
            # the videofile was not yet processed
            logger.info('Determining information by iterating through video '
                        '`%s` and store it in cache `%s`',
                        video_file, cache_file)
            # run ffprobe and fetch its output from the command line
            output = subprocess.check_output(cmd)
            # store result in the cache
            ffprobe_cache[video_file] = output
            
        else:
            logger.info('Loaded information about video `%s` from cache `%s`',
                        video_file, cache_file)
            
    else:
        # run ffprobe and fetch its output from the command line
        logger.info('Determining information by iterating through video '
                    '`%s`' % video_file)
        output = subprocess.check_output(cmd)

    # parse the json output
    infos = json.loads(output)

    if print_infos:
        print infos

    # select the first stream
    infos = infos["streams"][0]
    
    # add synonyms
    fps_e, fps_d = infos['r_frame_rate'].split('/')
    infos['video_size'] = (int(infos['width']), int(infos['height']))
    infos['video_fps'] = float(fps_e) / float(fps_d)
    infos['video_nframes'] = int(infos['nb_read_frames'])
    infos['video_duration'] = float(infos['duration'])
    
    return infos



def time_to_seconds(time):
    """ Will convert any time into seconds.
    Here are the accepted formats:
    
    >>> time_to_seconds(15.4) -> 15.4 # seconds
    >>> time_to_seconds((1, 21.5)) -> 81.5 # (min,sec)
    >>> time_to_seconds((1, 1, 2)) -> 3662 # (hr, min, sec)
    >>> time_to_seconds('01:01:33.5') -> 3693.5  #(hr,min,sec)
    >>> time_to_seconds('01:01:33.045') -> 3693.045
    >>> time_to_seconds('01:01:33,5') # comma works too
    """

    if isinstance(time, basestring):
        if (',' not in time) and ('.' not in time):
            time = time + '.0'
        expr = r"(\d+):(\d+):(\d+)[,|.](\d+)"
        finds = re.findall(expr, time)[0]
        nums = list( map(float, finds) )
        return (3600*int(finds[0])
                + 60*int(finds[1])
                + int(finds[2])
                + nums[3]/(10**len(finds[3])))
    
    elif isinstance(time, tuple):
        if len(time) == 3:
            hr, mn, sec = time
        elif len(time) == 2:
            hr, mn, sec = 0, time[0], time[1]    
        return 3600*hr + 60*mn + sec
    
    else:
        return time
    