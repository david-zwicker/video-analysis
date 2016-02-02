#!/usr/bin/env python2
'''
Created on Oct 18, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import datetime
import logging

import numpy as np

from ...simple import load_result_file
from video.filters import FilterCrop
from video.io import VideoComposer
from video.analysis.shapes import Rectangle
from utils.math import contiguous_true_regions
from utils.misc import display_progress



def make_underground_video(result_file, output_video=None, display='time',
                           scale_bar=True, min_duration=60, blank_duration=5):
    """ main routine of the program
    `result_file` is the file where the results from the video analysis are
        stored. This is usually a *.yaml file
    `output_video` denotes the filename where the result video should be
        written to.
    `display` determines what information is displayed. Possible values are
        'time' to display real time or 'frame' to display the frame number. All
         other values are ignored
    `scale_bar` determines whether a scale bar is shown
    `min_duration` determines how many frames the mouse has to be below ground
        for the bout to be included in the video
    `blank_duration` determines who many white frames are displayed between
        time intervals where the mouse is underground
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
        output_video = analyzer.get_filename('underground.mov', 'results')
    
    logging.info('Write output to `%s`', output_video)
    
    video_codec = analyzer.params['output/video/codec']
    video_bitrate = analyzer.params['output/video/bitrate']
    fps = video_input.fps
    video_output = VideoComposer(
        output_video, size=video_input.size, fps=fps, is_color=False,
        codec=video_codec, bitrate=video_bitrate,
    )
    
    # create blank frame with mean color of video
    blank_frame = np.full(video_input.shape[1:], video_input[0].mean(),
                          dtype=np.uint8)
    
    # time label position
    label_pos = video_input.width // 2, 30

    # calculate size of scale bar and the position of its label
    pixel_size_cm = analyzer.data['pass2/pixel_size_cm']
    scale_bar_size_cm = 10
    scale_bar_size_px = np.round(scale_bar_size_cm / pixel_size_cm)
    scale_bar_rect = Rectangle(30, 50, scale_bar_size_px, 5)
    scale_bar_pos = (30 + scale_bar_size_px//2, 30)

    # find time periods where the mouse is underground long enough
    bouts = contiguous_true_regions(mouse_ground_dists < 0) - frame_offset
    for start, finish in display_progress(bouts):
        
        duration = finish - start + 1 
        
        # check whether bout is long enough
        if duration < min_duration:
            continue
        
        if video_output.frames_written > 1:
            # write blank frame
            for _ in xrange(blank_duration):
                video_output.set_frame(blank_frame)
                
        video_bout = video_input[start:finish + 1]
        iterator = display_progress(enumerate(video_bout, start), duration)
        
        for frame_id, frame in iterator:
            video_output.set_frame(frame, copy=True)

            if scale_bar:
                # show a scale bar
                video_output.add_rectangle(scale_bar_rect, width=-1)
                video_output.add_text(str('%g cm' % scale_bar_size_cm),
                                      scale_bar_pos, color='w',
                                      anchor='upper center')

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
        