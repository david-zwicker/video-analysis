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


class FFMPEG_VideoWriter:
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
      Size (height, width) of the output video in pixels.
      
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
        
    def __init__(self, filename, size, fps, codec="libx264", is_color=True, bitrate=None):

        self.filename = filename
        self.codec = codec
        self.ext = self.filename.split(".")[-1]

        # determine whether debugging is requested
        debug = (logging.getLogger().getEffectiveLevel() <= logging.DEBUG)

        # build the ffmpeg command
        cmd = (
            [FFMPEG_BINARY, '-y',
            "-loglevel", "info" if debug else "error",
            "-f", 'rawvideo',
            "-vcodec","rawvideo",
            '-s', "%dx%d" % (size[1], size[0]), # ffmpeg expects width, height
            '-pix_fmt', "rgb24" if is_color else "gray",
            '-r', "%.02f" % fps,
            '-i', '-', '-an',
            '-vcodec', codec]
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
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=DEVNULL)

        
    def write_frame(self,img_array):
        """ Writes a single frame in the file """
        
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
        self.proc.stdin.close()
        if self.proc.stderr is not None:
            self.proc.stderr.close()
        self.proc.wait()
        
        del self.proc
    
    def __enter__(self):
        return self
        
    def __exit__(self, e_type, e_value, e_traceback):
        self.close()



def write_video_ffmpeg(video, filename, codec="libx264", bitrate=None):
    """
    Saves the video to the file indicated by filename.
    """
        
    logging.info('Start writing video with format `%s`', codec)
    
    # start ffmpeg and add the individual frames
    with FFMPEG_VideoWriter(filename, video.size, video.fps,
                            is_color=video.is_color, codec=codec,
                            bitrate=bitrate) as writer:
                 
        # write out all individual frames
        for frame in video:
            # convert the data to uint8 before writing it out
            writer.write_frame(frame.astype("uint8"))
            
    logging.info('Wrote video to file `%s`', filename)
    