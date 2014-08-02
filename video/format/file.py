'''
Created on Jul 31, 2014

@author: zwicker

This package provides class definitions for describing videos
that are based on a single file or on several files
'''

from __future__ import division

import logging
import itertools

from .base import VideoBase
from .backend_opencv import VideoOpenCV


# set default file handler
VideoFile = VideoOpenCV

class VideoStack(VideoBase):
    """
    Class handling a movie distributed over several files.
    The filenames must contain consecutive numbers
    """ 
    
    def __init__(self, filename_scheme='%d', index_start=1, index_end=None):
        
        # initialize the list containing all the files
        self.movies = []
        # register at what frame_count the video start
        self.offsets = []
        # internal pointer to the current movie from which to take a frame
        self._movie_pos = 0
        
        # determine over which indices we have to iterate
        if index_end is None:
            indices = itertools.count(index_start)
        else:
            indices = xrange(index_start, index_end+1)

        frame_count = 0
        last_movie = None
        for index in indices:
            
            # try to load the movie with given index
            try:
                movie = VideoFile(filename_scheme % index)
            except ValueError:
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
            self.offsets.append(frame_count)  
            frame_count += movie.frame_count

            # save the movie in the list
            self.movies.append(movie)
            
            logging.info('Found movie `%s`', movie.filename)
                        
        super(VideoStack, self).__init__(size=movie.size, frame_count=frame_count,
                                         fps=movie.fps, is_color=movie.is_color)


    def set_frame_pos(self, index):
        """ sets the 0-based index of the next frame """
        # set the frame position 
        super(VideoStack, self).set_frame_pos(index)
        
        # identify the movie that frame belongs to
        for self._movie_pos, movie_start in enumerate(self.offsets):
            if index < movie_start:
                break 


    def __iter__(self):
        """ initializes the iterator """
        # rewind all movies
        for movie in self.movies:
            movie.set_frame_pos(0)
        self._movie_pos = 0
        
        return self

            
    def next(self):
        """ returns the next frame in the video stack """
        
        # iterate until all movies are exhausted
        while True:
            try:
                # return next frame
                frame = self.movies[self._movie_pos].next()
                break
            except StopIteration:
                # if movie is exhausted, step to next movie
                self._movie_pos += 1
            except IndexError:
                # if the next movie does not exist, stop the iteration
                raise StopIteration
        
        # step to the next frame
        self._frame_pos += 1
        return frame
    

    def get_frame(self, index):
        """ returns a specific frame identified by its index """
        
        self.set_frame_pos(index)
        return self.next()
            
