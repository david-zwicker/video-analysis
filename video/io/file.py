'''
Created on Jul 31, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

This package provides class definitions for describing videos
that are based on a single file or on several files.
'''

from __future__ import division

import os
import glob
import itertools
import logging

from .base import VideoBase
from .backend_opencv import (show_video_opencv, VideoWriterOpenCV, VideoOpenCV,
                             VideoImageStackOpenCV)
from .backend_ffmpeg import (FFMPEG_BINARY, VideoFFmpeg, VideoWriterFFmpeg)

logger = logging.getLogger('video.io')

# set default handlers
show_video = show_video_opencv
VideoImageStack = VideoImageStackOpenCV

if FFMPEG_BINARY is not None:
    VideoFile = VideoFFmpeg
    VideoFileWriter = VideoWriterFFmpeg
else:
    VideoFile = VideoOpenCV
    VideoFileWriter = VideoWriterOpenCV



def load_any_video(video_filename_pattern, parameters=None):
    """ loads either a video file or a video file stack, depending
    on the video_filename_pattern supplied """
    if any(c in video_filename_pattern for c in r'*?%'):
        # contains placeholder => load multiple videos
        return VideoFileStack(video_filename_pattern, keep_files_open=False,
                              parameters=parameters)
    else:
        # no placeholder => load single video
        return VideoFile(video_filename_pattern, parameters=parameters)



def write_video(video, filename, **kwargs):
    """
    Saves the video to the file indicated by filename.
    The extra arguments determine the codec used and similar parameters.
    The accepted values depend on the backend chosen for the video writer.
    """
        
    # initialize the video writer
    with VideoFileWriter(filename, size=video.size, fps=video.fps,
                         is_color=video.is_color, **kwargs) as writer:
                 
        # write out all individual frames
        for frame in video:
            # convert the data to uint8 before writing it out
            writer.write_frame(frame)
    


class VideoFileStack(VideoBase):
    """
    Class handling a video distributed over several files.
    The filenames must contain consecutive numbers
    """ 
    
    def __init__(self, filename_scheme='%d', index_start=1, index_end=None,
                 video_file_class=VideoFile, keep_files_open=True,
                 parameters=None):
        """
        initialize the VideoFileStack.
        
        A list of videos is found using a filename pattern, where two
        alternative patterns are supported:
            1) Using linux globs, i.e. placeholders * and ? in normal files
            2) Using enumeration, where %d is replaced by consecutive integers
        For the second method, the start and end of the running index can
        be determined using index_start and index_end.
        
        video_file_class determines the class with which videos are loaded
        keep_files_open determines whether all files are kept open at all times.
            Otherwise only the file that is currently needed is opened. This
            can result in severe performance penalties if frames are accessed
            randomly. Conversely, keeping only one file open at a time can
            increase robustness.
        """
        
        # initialize the list containing all the files
        self._videos = []
        # register at what frame_count the video start
        self._offsets = []
        # internal pointer to the current video from which to take a frame
        self._video_pos = 0
        
        # get parameters from video_file_class and the ones given here 
        self.parameters = video_file_class.parameters_default.copy()
        if parameters:
            self.parameters.update(parameters)
        
        self.keep_files_open = keep_files_open
        
        # find all files that have to be considered
        if '*' in filename_scheme or '?' in filename_scheme:
            logger.debug('Using glob module to locate files.')
            filenames = sorted(glob.glob(filename_scheme))
            
        elif r'%' in filename_scheme:
            logger.debug('Iterating over possible filenames to find videos.')
        
            # determine over which indices we have to iterate
            if index_end is None:
                indices = itertools.count(index_start)
            else:
                indices = xrange(index_start, index_end+1)

            filenames = []                
            for index in indices:
                filename = filename_scheme % index
    
                # append filename to list if file is readable
                if os.path.isfile(filename) and os.access(filename, os.R_OK):
                    filenames.append(filename)
                else:
                    break

        else:
            logger.warn('It seems as the filename scheme refers to a single '
                        'file.')
            filenames = [filename_scheme]

        if not filenames:
            raise IOError('Could not find any files matching the pattern `%s`'
                          % filename_scheme)

        # load all the files that have been found
        self.seekable = True
        frame_count = 0
        last_video = None
        for filename in filenames:
            
            # try to load the video with given index
            try:
                video = video_file_class(filename, parameters=self.parameters)
            except IOError:
                raise IOError('Could not read video `%s`' % filename)
                continue
            
            # compare its format to the previous videos
            if last_video:
                if video.fps != last_video.fps:
                    raise ValueError('The frame rates of two videos differ')
                if video.size != last_video.size:
                    raise ValueError('The sizes of two videos differ')
                if video.is_color != last_video.is_color:
                    raise ValueError('The color formats of two videos differ')
            
            # set parameters of the video
            if self.parameters:
                video.parameters = self.parameters
            
            # calculate at which frame this video starts
            self._offsets.append(frame_count)  
            frame_count += video.frame_count

            # save the video in the list
            self._videos.append(video)
            
            # check whether this video is seekable
            if not video.seekable:
                self.seekable = False
            
            logger.info('Found video `%s`', video.filename)

            if not self.keep_files_open:
                video.close()
                        
        if not self._videos:
            raise RuntimeError('Could not load any videos')
                        
        super(VideoFileStack, self).__init__(size=video.size,
                                             frame_count=frame_count,
                                             fps=video.fps,
                                             is_color=video.is_color)


    @property
    def filecount(self):
        return len(self._videos)
    

    def get_property_list(self):
        return super(VideoFileStack, self).get_property_list() \
               + ('filecount=%s' % self.filecount,)


    def get_video_index(self, frame_index):
        """ returns the video and local frame_index to which a certain frame
        belongs """
        
        for video_index, video_start in enumerate(self._offsets):
            if frame_index < video_start:
                video_index -= 1
                break
        
        return video_index, frame_index - self._offsets[video_index] 
    

    def set_frame_pos(self, index):
        """ sets the 0-based index of the next frame.
        This opens a potentially closed video and keeps it open afterwards.
        This also rewinds all successive open videos.
        """
        if index < 0:
            index += self.frame_count

        # set the frame position 
        super(VideoFileStack, self).set_frame_pos(index)
        
        # identify the video that frame belongs to
        self._video_pos, frame_index = self.get_video_index(index)
        
        video = self._videos[self._video_pos]
        if video.closed:
            video.open()
        video.set_frame_pos(frame_index)

        # rewind all subsequent _videos, because we cannot start iterating 
        # from the current position of the video
        for video in self._videos[self._video_pos + 1:]:
            if not video.closed:
                video.set_frame_pos(0)

            
    def get_next_frame(self):
        """ returns the next frame in the video stack """
        
        # iterate until all _videos are exhausted
        while True:
            try:
                # return next frame
                video = self._videos[self._video_pos]
                if video.closed:
                    video.open()
                frame = video.get_next_frame()
                break
            
            except StopIteration:
                # if video is exhausted, step to next video
                if not self.keep_files_open:    
                    self._videos[self._video_pos].close()
                self._video_pos += 1
                
            except IndexError:
                # if the next video does not exist, stop the iteration
                raise StopIteration
        
        # step to the next frame
        self._frame_pos += 1
        return frame
    

    def get_frame(self, index):
        """ returns a specific frame identified by its index """
        if index < 0:
            index += self.frame_count

        video_index, frame_index = self.get_video_index(index)
        video = self._videos[video_index]
        if video.closed:
            video.open()
        frame = video.get_frame(frame_index)
        if not self.keep_files_open:
            video.close()
        return frame
            

    def close(self):
        for video in self._videos:
            video.close()
