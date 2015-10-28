#!/usr/bin/env python2
'''
Created on Oct 18, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import datetime
import logging
import sys
import os

import numpy as np

# add the root of the video-analysis project to the path
this_path = os.path.dirname(__file__)
video_analysis_path = os.path.join(this_path, '..', '..')
sys.path.append(video_analysis_path)

from ..simple import load_result_file
from video.filters import FilterCrop
from video.io import VideoComposer
from utils.misc import display_progress



def make_underground_video(result_file, output_video=None, display='time',
                           blank_duration=5, blank_wait=50):
    """ main routine of the program
    `result_file` is the file where the results from the video analysis are
        stored. This is usually a *.yaml file
    `output_video` denotes the filename where the result video should be
        written to.
    `display` determines what information is displayed. Possible values are
        'time' to display real time or 'frame' to display the frame number. All
         other values are ignored
    `blank_duration` determines who many white frames are displayed between
        time intervals where the mouse is underground
    `blank_wait` determines how many frames the mouse has to be above ground for
        it to trigger the insertion of white frames
    """
    logging.info('Analyze video `%s`', result_file)
    
    # load the respective result file 
    analyzer = load_result_file(result_file)
    
    # load the original video
    video_info = analyzer.load_video()
    video_input = analyzer.video
    frame_offset = video_info['frames'][0]
    
    # crop the video to the cage
    cropping_cage = analyzer.data['pass1/video/cropping_cage']
    if cropping_cage:
        video_input = FilterCrop(video_input, rect=cropping_cage)
    
    # get the distance of the mouse to the ground
    mouse_ground_dists = analyzer.get_mouse_track_data('ground_dist',
                                                       night_only=False)
    
    # check whether we have enough information and issue a message otherwise
    if np.all(np.isnan(mouse_ground_dists)):
        raise RuntimeError('The distance of the mouse to the ground is not '
                           'available. Either the third pass has not finished '
                           'yet or there was a general problem with the video '
                           'analysis.')
    
    # create output video
    if output_video is None:
        output_video = analyzer.get_filename('underground', 'results')
    
    video_codec = analyzer.params['output/video/codec']
    video_bitrate = analyzer.params['output/video/bitrate']
    fps = video_input.fps
    video_output = VideoComposer(
        output_video, size=video_input.size, fps=fps, is_color=False,
        codec=video_codec, bitrate=video_bitrate,
    )
    blank_frame = np.full(video_input.shape[1:], 255, dtype=np.uint8)
    
    # time label position
    label_pos = video_input.width // 2, 30

    iterator = display_progress(video_input)
    frame_last = None
    for frame_id, frame in enumerate(iterator, frame_offset):
        #print frame_id, video_input.get_frame_pos()
        try:
            mouse_dist = mouse_ground_dists[frame_id]
        except IndexError:
            break

        if mouse_dist < 0:
            # add white frames if the mouse was above ground for a while
            if frame_last is not None and frame_last < frame_id - blank_wait:
                for _ in xrange(blank_duration):
                    video_output.set_frame(blank_frame)            
            
            video_output.set_frame(frame, copy=True)

            if display == 'time':
                # output time
                time_secs, time_frac = divmod(frame_id, fps)
                time_msecs = int(1000 * time_frac / fps)
                dt = datetime.timedelta(seconds=time_secs,
                                        milliseconds=time_msecs)
                video_output.add_text(str(dt), label_pos, color='w',
                                      anchor='upper center')
                
            elif display == 'frame':
                # output the frame
                video_output.add_text(str(frame_id), label_pos, color='w',
                                      anchor='upper center')
            
            frame_last = frame_id
    
    # show summary
    frames_total = video_info['frames'][1] - video_info['frames'][0]
    frames_written = video_output.frames_written
    logging.info('%d (%d%%) of %d frames written', frames_written,
                 100 * frames_written // frames_total, frames_total)
    
    # close and finalize video
    try:
        video_output.close()
    except IOError:
        logging.exception('Error while writing out the debug video `%s`',
                          video_output) 
    


if __name__ == '__main__':
    make_underground_video(*sys.argv)
    
    
