'''
Created on Jul 31, 2014

@author: zwicker

This package provides class definitions for referencing a single video file.

This code has been modified from the project moviepy, which is released under
the MIT license at github:
https://github.com/Zulko/moviepy/blob/master/moviepy/video/io/ffmpeg_writer.py

The MIT license text is included in the present package in the main README file
'''

from __future__ import division

import logging
import subprocess

import numpy as np

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


# search for the ffmpeg command
if try_cmd(['ffmpeg']):
    FFMPEG_BINARY = 'ffmpeg'
elif try_cmd(['ffmpeg.exe']):
    FFMPEG_BINARY = 'ffmpeg.exe'
else:
    FFMPEG_BINARY = None
    print("FFMPEG binary not found. Functions relying on this will not be available.")


class VideoWriterFFMPEG(object):
    """ A class for FFMPEG-based video writing.
    
    A class to write videos using ffmpeg. ffmpeg will write in a large
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
      FFMPEG codec. It seems that in terms of quality the hierarchy is
      'rawvideo' = 'png' > 'mpeg4' > 'libx264'
      'png' manages the same lossless quality as 'rawvideo' but yields
      smaller files. Type ``ffmpeg -codecs`` in a terminal to get a list
      of accepted codecs.

      Note for default 'libx264': by default the pixel format yuv420p
      is used. If the video dimensions are not both even (e.g. 720x405)
      another pixel format is used, and this can cause problem in some
      video readers.

    bitrate
      Only relevant for codecs which accept a bitrate. "5000k" offers
      nice results in general.
    
    """
        
    def __init__(self, filename, size, fps, is_color=True, codec="libx264", bitrate=None):

        self.filename = filename
        self.codec = codec
        self.ext = self.filename.split(".")[-1]
        self.size = size
        self.is_color = is_color        

        if size[0]%2 != 0 or size[1]%2 != 0:
            raise ValueError('Both dimensions of the video must be even for '
                             'the video codec to work properly')

        # build the ffmpeg command
        cmd = (
            [FFMPEG_BINARY, '-y',
            "-loglevel", "error", #"info" if verbose() else "error",
            "-f", 'rawvideo',
            "-vcodec","rawvideo",
            '-s', "%dx%d" % size,
            '-pix_fmt', "rgb24" if is_color else "gray",
            '-r', "%.02f" % fps,
            '-i', '-',
            '-an'] # no audio
            + ([] if (codec is None) else ['-vcodec', codec])
            + ([] if (bitrate is None) else ['-b', bitrate])

            # http://trac.ffmpeg.org/ticket/658
            + (['-pix_fmt', 'yuv420p']
                  if ((codec == 'libx264') and
                     (size[0]%2 == 0) and
                     (size[1]%2 == 0))
                     
               else [])

            + ['-r', "%d" % fps, filename]
        )

        # start ffmpeg, which should wait for input
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=DEVNULL, stderr=subprocess.PIPE)

        logging.info('Start writing video `%s` with codec `%s`', filename, codec)

    
    @property
    def shape(self):
        """ returns the shape of the data describing the movie """
        shape = (self.size[1], self.size[0])
        if self.is_color:
            shape += (3,)
        return shape
    
        
    def write_frame(self, img_array):
        """ Writes a single frame in the file """
        
        img_array = img_array.astype("uint8")   
        
        if self.is_color and img_array.ndim == 2:            
            img_array = img_array[:, :, None]*np.ones((1, 1, 3), np.uint8)
            
        try:
            self.proc.stdin.write(img_array.tostring())
        except IOError as err:
            ffmpeg_error = self.proc.stderr.read()
            error = (str(err) + ("\n\nMoviePy error: FFMPEG encountered "
                     "the following error while writing file %s:" % self.filename
                     + "\n\n" + ffmpeg_error))

            if "Unknown encoder" in ffmpeg_error:                
                error = error+("\n\nThe video export "
                  "failed because FFMPEG didn't find the specified "
                  "codec for video encoding (%s). Please install "
                  "this codec or use a different codec") % (self.codec)
            
            elif "incorrect codec parameters ?" in ffmpeg_error:
                error = error + ("\n\nThe video export "
                  "failed, possibly because the codec specified for "
                  "the video (%s) is not compatible with the given "
                  "extension (%s). Please specify a valid 'codec' "
                  "argument in write_videofile. This would be 'libx264' "
                  "or 'mpeg4' for mp4, 'libtheora' for ogv, 'libvpx' "
                  "for webm.") % (self.codec, self.ext)

            elif  "encoder setup failed":
                error = error + ("\n\nThe video export "
                  "failed, possibly because the bitrate you specified "
                  "was too high or too low for the video codec.")
            
            raise IOError(error)
        
        
    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            if self.proc.stderr is not None:
                self.proc.stderr.close()
            self.proc.wait()
        
            logging.info('Wrote video to file `%s`', self.filename)
            
            self.proc = None

    
    
    def __enter__(self):
        return self
    
        
    def __exit__(self, e_type, e_value, e_traceback):
        self.close()

    
    def __del__(self):
        self.close()

