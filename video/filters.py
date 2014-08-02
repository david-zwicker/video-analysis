'''
Created on Aug 1, 2014

@author: zwicker

Filter are generators that take a video as an input and return filtered
output as one iterates over them
'''


def crop(source, rect):
    """ crops the video to the given rect=(left, top, right, bottom) """ 
    for frame in source:
        yield frame[rect[0]:rect[2], rect[1]:rect[3], :]


def time_difference(source):
    """ returns the differences between consecutive frames """ 
    
    # get the iterator from the video
    data = iter(source)
    
    # iterate through all frames and yield the difference
    last_frame = data.next()
    while data:
        frame = data.next()
        yield frame - last_frame
        last_frame = frame
        
        
def normalize_brightness(source):
    """
    adjusts individual frames such that their brightness corresponds to
    the initial frame
    """ 
    raise NotImplementedError

