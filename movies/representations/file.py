'''
Created on Jul 31, 2014

@author: zwicker

Package provides an abstract base class to define an interface and common
functions for video handling. Concrete implementations are collected in the
backend subpackage.
'''

from __future__ import division


from .base import MovieBase
from .opencv import OpenCVMovie


# set default file handler
MovieFile = OpenCVMovie


class MovieBatch(MovieBase):
    """
    Class handling a movie distributed over several files.
    The filenames must contain consecutive numbers
    """ 
    
    def __init__(self, filename_scheme='%d'):
        
        # initialize the list containing all the files
        self.movies = []
        # register at what frame_count the movies start
        self.offsets = []
        
        frame_count = 0
        index = 0
        last_movie = None
        try:
            while True:
                # load the movie
                movie = MovieFile(filename_scheme % index)
                
                # compare it to the previous movie
                if last_movie:
                    if movie.fps != last_movie.fps:
                        raise ValueError('The FPS value of two movies does not agree')
                    if movie.size != last_movie.size:
                        raise ValueError('The size of two movies does not agree')
                
                # calculate at which frame this movie starts
                self.offsets.append(frame_count)  
                frame_count += movie.frame_count

                # save the movie in the list
                self.movies.append(movie)
                
        except IOError:
            # assume that there are no more files and we are done registering the movies
            pass
        
        super(MovieBatch, self).__init__(size=movie.size, frame_count=frame_count, fps=movie.fps)


    def get_frame_raw(self, index):
        """ returns a specific frame identified by its index """
        
        if index >= self.frame_count:
            raise StopIteration
        
        # find the id of the movie in which the frame resides 
        for movie_id, movie_start in enumerate(self.offsets):
            if index < movie_start:
                break
        
        # load the movie and get the respective frame 
        movie = self.movies[movie_id]
        frame = movie.get_frame_raw(index - movie_start)
        
        # set internal pointer to next frame
        self._frame_pos = index + 1
        
        return frame
            

    def get_next_frame_raw(self):
        """ returns the next frame """

        # this also sets the internal pointer to the next frame
        frame = self.get_frame_raw(self._frame_pos)
        return frame
