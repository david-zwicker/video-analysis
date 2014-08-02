'''
Created on Jul 31, 2014

@author: zwicker

This package provides class definitions for describing videos
that are based on a single file or on several files
'''

from __future__ import division


from .base import VideoBase
from .backend_opencv import VideoOpenCV


# set default file handler
VideoFile = VideoOpenCV

class VideoStack(VideoBase):
    """
    Class handling a movie distributed over several files.
    The filenames must contain consecutive numbers
    """ 
    
    def __init__(self, filename_scheme='%d'):
        
        # initialize the list containing all the files
        self.movies = []
        # register at what frame_count the video start
        self.offsets = []
        
        frame_count = 0
        index = 0
        last_movie = None
        try:
            while True:
                # load the movie
                movie = VideoFile(filename_scheme % index)
                
                # compare it to the previous movie
                if last_movie:
                    if movie.fps != last_movie.fps:
                        raise ValueError('The FPS value of two video does not agree')
                    if movie.size != last_movie.size:
                        raise ValueError('The size of two video does not agree')
                
                # calculate at which frame this movie starts
                self.offsets.append(frame_count)  
                frame_count += movie.frame_count

                # save the movie in the list
                self.movies.append(movie)
                
        except IOError:
            # assume that there are no more files and we are done registering the video
            pass
        
        super(VideoCollection, self).__init__(size=movie.size, frame_count=frame_count, fps=movie.fps)


    def get_frame(self, index):
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
            
