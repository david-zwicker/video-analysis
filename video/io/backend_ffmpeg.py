'''
Created on Jul 31, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

This package provides class definitions for referencing a single video file.

This code has been modified from the project moviepy, which is released under
the MIT license at github:
https://github.com/Zulko/moviepy/blob/master/moviepy/video/io

The MIT license text is included in the present package in the file
/lib/LICENSE_MIT.txt
'''

from __future__ import division

import re
import logging
import subprocess

import numpy as np

from .base import VideoBase

logger = logging.getLogger('video.io')

# find the file handle to /dev/null to dumb strings
try:
    from subprocess import DEVNULL # py3k
except ImportError:
    import os
    DEVNULL = open(os.devnull, 'wb')


def try_cmd(cmd):
    """ helper function checking whether a command runs successful """    
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        proc.communicate()
    except:
        return False
    else:
        return True
from subprocess import check_output


# search for the FFmpeg command
if try_cmd(['ffmpeg']):
    FFMPEG_BINARY = 'ffmpeg'
    logger.debug('Found ffmpeg at: %s', check_output(['which', 'ffmpeg']).strip())
elif try_cmd(['ffmpeg.exe']):
    FFMPEG_BINARY = 'ffmpeg.exe'
    logger.debug('Found ffmpeg.exe.')
else:
    FFMPEG_BINARY = None
    logger.warn("ffmpeg binary not found. Functions relying on this will not be available.")



class VideoFFmpeg(VideoBase):
    """ Class handling a single movie file using FFmpeg
    """ 

    def __init__(self, filename, bufsize=None, pix_fmt="rgb24"):

        self.filename = filename
        
        # get information about the frame using FFmpeg
        infos = ffmpeg_parse_infos(filename)
        self.duration = infos['video_duration']
        self.FFmpeg_duration = infos['duration']

        self.infos = infos

        self.pix_fmt = pix_fmt
        if pix_fmt == 'rgba':
            self.depth = 4
        elif pix_fmt == 'rgb24':
            self.depth = 3
        else:
            raise ValueError('Unsupported pixel format `%s`' % pix_fmt)

        if bufsize is None:
            w, h = infos['video_size']
            bufsize = self.depth * w * h + 100

        self.bufsize = bufsize
        self.proc = None
        self.open()

        self.lastread = None
        
        super(VideoFFmpeg, self).__init__(size=tuple(infos['video_size']),
                                          frame_count=infos['video_nframes'],
                                          fps=infos['video_fps'], is_color=True)

        logger.debug('Initialized video `%s` with %d frames using OpenCV',
                     filename, infos['video_nframes'])


    def print_infos(self):
        """ print information about the video file """
        ffmpeg_parse_infos(self.filename, print_infos=True)


    @property
    def closed(self):
        return self.proc is None


    def open(self, index=0):
        """ Opens the file, creates the pipe. """
        self.close() # close if anything was opened
        
        if index != 0:
            # -0.1 is necessary to prevent rounding errors
            starttime = (index - 0.1)/self.fps
            offset = min(1, starttime)
            # this rewinds to the previous keyframe and than progresses slowly
            i_arg = ['-ss', "%.03f" % (starttime - offset),
                     '-i', self.filename,
                     '-ss', "%.03f" % offset]
            logger.debug('Seek video to frame %d (=%.03f sec)' % (index, starttime))
            
        else:
            i_arg = ['-i', self.filename]
        
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
        self._frame_pos = index


    def set_frame_pos(self, index):
        """ sets the video to position index """
        if index != self._frame_pos:
            if (index < self._frame_pos) or (index > self._frame_pos + 100):
                self.open(index)
            else:
                skip_frames = index - self._frame_pos
                w, h = self.size
                for _ in xrange(skip_frames):
                    self.proc.stdout.read(self.depth*w*h)
                    self.proc.stdout.flush()
                self._frame_pos = index
                

    def get_next_frame(self):
        """ retrieve the next frame from the video """
        w, h = self.size
        nbytes = self.depth*w*h

        s = self.proc.stdout.read(nbytes)

        if len(s) != nbytes:
            # frame_count is a rather crude estimate
            # We thus stop the iteration, when we think we are close to the end 
            if self._frame_pos >= 0.99*self.frame_count:
                raise StopIteration
            
            logger.warn("Warning: in file %s, %d bytes wanted but %d bytes read, "
                        "at frame %d/%d, at time %.02f/%.02f sec. "
                        "Using the last valid frame instead." %
                        (self.filename, nbytes, len(s),
                         self._frame_pos, self.frame_count,
                         self._frame_pos/self.fps, self.duration))
            
            if not hasattr(self, 'lastread'):
                raise IOError("Failed to read the first frame of video file %s. "
                              "That might mean that the file is corrupted. That "
                              "may also mean that you are using a deprecated "
                              "version of FFmpeg." % self.filename)

            result = self.lastread

        else:
            # frame has been obtained properly
            result = np.frombuffer(s, dtype='uint8').reshape((h, w, len(s)//(w*h)))
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
            self.proc.terminate()
            self.proc.stdout.close()
            self.proc.stderr.close()
            self.proc = None
    
    
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
                 bitrate=None, debug=False):

        self.filename = filename
        self.codec = codec
        self.ext = self.filename.split(".")[-1]
        self.size = size
        self.is_color = is_color
        self.debug = debug        

        if size[0]%2 != 0 or size[1]%2 != 0:
            raise ValueError('Both dimensions of the video must be even for '
                             'the video codec to work properly')

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
            + ([] if (codec is None) else ['-vcodec', codec])
            + ([] if (bitrate is None) else ['-b', bitrate])

            # http://trac.FFmpeg.org/ticket/658
            + (['-pix_fmt', 'yuv420p']
                  if ((codec == 'libx264') and
                     (size[0]%2 == 0) and
                     (size[1]%2 == 0))
                     
               else [])

            + ['-r', "%d" % fps, filename]
        )
        
        # start FFmpeg, which should wait for input
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=DEVNULL, stderr=subprocess.PIPE)

        logger.info('Start writing video `%s` with codec `%s`', filename, codec)

    
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
            
            raise IOError(error)
        
        # check for extra output in debug mode
        if self.debug:
            stderr_read = os.read(self.proc.stderr.fileno(), 1024)
            FFmpeg_output = stderr_read 
            while len(stderr_read) == 1024:
                stderr_read = os.read(self.proc.stderr.fileno(), 1024)
                FFmpeg_output += stderr_read
            logger.info(FFmpeg_output)
        
        
    def close(self):
        """ finishes the process, which should also make the video available """
        if self.proc is not None:
            self.proc.communicate()
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
                      "Please check that you entered the correct "
                      "path.\n"
                      "Here are the file information returned by FFmpeg:\n\n%s"
                      % (filename, infos))
    
    # get duration (in seconds)
    result = {}
    try:
        keyword = ('frame=' if is_GIF else 'Duration: ')
        line = [l for l in lines if keyword in l][0]
        match = re.findall("([0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9])", line)[0]
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
        # Current policy: Trust tbr first, then fps. If result is near from x*1000/1001
        # where x is 23,24,25,50, replace by x*1000/1001 (very common case for the fps).
        
        try:
            match = re.search("( [0-9]*.| )[0-9]* tbr", line)
            tbr = float(line[match.start():match.end()].split(' ')[1])
            result['video_fps'] = tbr

        except:
            match = re.search("( [0-9]*.| )[0-9]* fps", line)
            result['video_fps'] = float(line[match.start():match.end()].split(' ')[1])

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