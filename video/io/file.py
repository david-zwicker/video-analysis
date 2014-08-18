'''
Created on Jul 31, 2014

@author: zwicker

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
from .backend_ffmpeg import (FFMPEG_BINARY, VideoWriterFFMPEG)

# set default handlers
show_video = show_video_opencv
VideoFile = VideoOpenCV
VideoImageStack = VideoImageStackOpenCV

if FFMPEG_BINARY is not None:
    VideoFileWriter = VideoWriterFFMPEG
else:
    VideoFileWriter = VideoWriterOpenCV



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
    Class handling a movie distributed over several files.
    The filenames must contain consecutive numbers
    """ 
    
    def __init__(self, filename_scheme='%d', index_start=1, index_end=None, video_file_class=VideoFile):
        
        # initialize the list containing all the files
        self._movies = []
        # register at what frame_count the video start
        self._offsets = []
        # internal pointer to the current movie from which to take a frame
        self._movie_pos = 0
        
        # find all files that have to be considered
        if '*' in filename_scheme or '?' in filename_scheme:
            logging.debug('Using glob module to locate files.')
            filenames = sorted(glob.glob(filename_scheme))
            
        elif r'%' in filename_scheme:
            logging.debug('Iterating over possible filenames to find movies.')
        
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
            logging.warn('It seems as the filename scheme refers to a single file.')
            filenames = [filename_scheme]

        if not filenames:
            raise IOError('Could not find any files matching the pattern `%s`' % filename_scheme)

        # load all the files that have been found
        frame_count = 0
        last_movie = None
        for filename in filenames:
            
            # try to load the movie with given index
            try:
                movie = video_file_class(filename)
            except IOError:
                break
            
            # compare its format to the previous movies
            if last_movie:
                if movie.fps != last_movie.fps:
                    raise ValueError('The FPS value of two videos does not agree')
                if movie.size != last_movie.size:
                    raise ValueError('The size of two videos does not agree')
                if movie.is_color != last_movie.is_color:
                    raise ValueError('The color format of two videos does not agree')
            
            # calculate at which frame this movie starts
            self._offsets.append(frame_count)  
            frame_count += movie.frame_count

            # save the movie in the list
            self._movies.append(movie)
            
            logging.info('Found movie `%s`', movie.filename)
                        
        super(VideoFileStack, self).__init__(size=movie.size, frame_count=frame_count,
                                             fps=movie.fps, is_color=movie.is_color)


    @property
    def filecount(self):
        return len(self._movies)
    

    def get_movie_index(self, frame_index):
        """ returns the movie and local frame_index to which a certain frame belongs """
        
        for movie_index, movie_start in enumerate(self._offsets):
            if frame_index < movie_start:
                movie_index -= 1
                break

        return movie_index, frame_index - self._offsets[movie_index] 
    

    def set_frame_pos(self, index):
        """ sets the 0-based index of the next frame """
        # set the frame position 
        super(VideoFileStack, self).set_frame_pos(index)
        
        # identify the movie that frame belongs to
        self._movie_pos, frame_index = self.get_movie_index(index)
        self._movies[self._movie_pos].set_frame_pos(frame_index)
        
        # rewind all subsequent _movies, because we cannot start iterating 
        # from the current position of the video
        for m in self._movies[self._movie_pos + 1:]:
            m.set_frame_pos(0)


    def _start_iterating(self):
        """ initializes the iterator """
        # reset internal movie index
        self._movie_pos = 0
        
        # rewind all movies
        for movie in self._movies:
            movie._start_iterating()
        
        super(VideoFileStack, self)._start_iterating()


    def _end_iterating(self):
        super(VideoFileStack, self)._end_iterating()

        for movie in self._movies:
            movie._end_iterating()
           
            
    def next(self):
        """ returns the next frame in the video stack """
        
        # iterate until all _movies are exhausted
        while True:
            try:
                # return next frame
                frame = self._movies[self._movie_pos].next()
                break
            
            except StopIteration:
                # if movie is exhausted, step to next movie
                self._movie_pos += 1
                
            except IndexError:
                # if the next movie does not exist, stop the iteration
                self._end_iterating()
                raise StopIteration
        
        # step to the next frame
        self._frame_pos += 1
        return frame
    

    def get_frame(self, index):
        """ returns a specific frame identified by its index """
        
        movie_index, frame_index = self.get_movie_index(index)
        return self._movies[movie_index].get_frame(frame_index)
            
