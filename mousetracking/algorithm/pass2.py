'''
Created on Aug 19, 2014

@author: zwicker

Module that contains the class responsible for the second pass of the algorithm
'''

from __future__ import division

import numpy as np

from .data_handler import DataHandler
from video.analysis import curves
from video.composer import VideoComposer
from video.filters import FilterCrop

import debug  # @UnusedImport


class SecondPass(DataHandler):
    """ class containing methods for the second pass """
    
    def __init__(self, name='', parameters=None, debug_output=None):
        super(SecondPass, self).__init__(name, parameters)
        self.initialize_parameters()
        self.params = self.data['parameters']
        self.log_event('Pass 2 - Started initializing the analysis.')

        self.debug_output = [] if debug_output is None else debug_output
        

    @classmethod
    def from_first_pass(cls, first_pass):
        """ create the object directly from the first pass """
        # create the data and copy the data from first_pass
        obj = cls(first_pass.name)
        obj.data = first_pass.data
        obj.params = first_pass.data['parameters']
        obj.tracks = first_pass.tracks
        # initialize parameters
        obj.initialize_parameters()
        return obj
    

    def process_data(self):
        """ do the second pass of the analysis """
        self.debug_setup()
        
        self.find_mouse_track()
        #self.smooth_ground_profile()
        #self.classify_mouse_track()
        #self.produce_video()
        
        self.debug_finalize()
        self.log_event('Pass 2 - Finished second pass.')
    
    
    def load_video(self, video=None, crop_video=True):
        """ load the video, potentially using a previously analyzed video """
        # load the video with the standard method
        super(SecondPass, self).load_video(video, crop_video=crop_video)
        if self.data['video/analyzed']:
            # apparently a video has already been analyzed
            # ==> we thus use the previously determined cage cropping rectangle
            crop_rect = self.data['video/analyzed/region_cage']
            self.video = FilterCrop(self.video, crop_rect)
        

    #===========================================================================
    # CONNECT TEMPORAL DATA -- TRACKING
    #===========================================================================

        
    def find_mouse_track(self):
        """ identifies the mouse trajectory by connecting object tracks.
        
        This function takes the tracks in 'pass1/objects/tracks', connects
        suitable parts, and interpolates gaps.
        """
        self.log_event('Pass 2 - Started identifying mouse trajectory.')
        # retrieve data
        tracks = self.data['pass1/objects/tracks']
        
        # find the longest track as a basis for finding the complete track
        core = max(tracks, key=lambda track: len(track))

        # find all tracks after this
        time_point = core.times[-1]
        tracks_after = [track
                        for track in tracks
                        if track.times[0] > time_point]
        # sort them according to their start time
        tracks_after = sorted(tracks_after, key=lambda track: track.times[0])


        def score_connection(left, right, segment_length):
            """ scoring function that defines how well the two tracks match """
            obj_l, obj_r = left.objects[-1], right.objects[0] #< the respective objects
            
            # calculate the distance between new and old objects
            dist_score = curves.point_distance(obj_l.pos, obj_r.pos)
            # normalize distance to the maximum speed
            dist_score /= self.params['mouse/max_speed']
            # calculate the difference of areas between new and old objects
            area_score = abs(obj_l.size - obj_r.size)/(obj_l.size + obj_r.size)

            # build a combined score from this
            alpha = self.params['objects/matching_weigth']
            score = alpha*dist_score + (1 - alpha)*area_score
            
            # score length of the new segment
            score *= np.log(1 + segment_length)  
            
            # decrease score with gap length
            score /= np.log(1 + right.times[0] - left.times[-1])
            
            return score
            
            
        result = [core] #< list of final tracks
        score = {} #< dictionary of scores for potential matches
        k = 0
        while k < len(tracks_after):
            track = tracks_after[k]
            score[k] = score_connection(result[-1], track, len(track))
            if score[k] > 1:
                print result[-1][-1], track[0]
            #print k, score[k],track
            k += 1
            
        return result
                

    #===========================================================================
    # DEBUGGING
    #===========================================================================


    def debug_setup(self):
        """ prepares everything for the debug output """
        self.debug['video.mark.text1'] = ''
        self.debug['video.mark.text2'] = ''

        # load parameters for video output        
        video_extension = self.params['output/video/extension']
        video_codec = self.params['output/video/codec']
        video_bitrate = self.params['output/video/bitrate']
        
        # set up the general video output, if requested
        if 'video' in self.debug_output or 'video.show' in self.debug_output:
            # initialize the writer for the debug video
            debug_file = self.get_filename('video' + video_extension, 'debug')
            self.debug['video'] = VideoComposer(debug_file, background_video=self.video,
                                                        is_color=True, codec=video_codec,
                                                        bitrate=video_bitrate)
        

    def debug_add_frame(self, frame):
        """ adds information of the current frame to the debug output """
        if 'video' in self.debug:
            # TODO: create meaningful debug output for the second pass 
            debug_video = self.debug['video']
            
            # plot the ground profile
            debug_video.add_polygon(self.ground, is_closed=False, mark_points=True, color='y')
        
            # indicate the currently active burrow shapes
            for burrow_track in self.result['burrows/data']:
                if burrow_track.last_seen > self.frame_id - self.params['burrows/adaptation_interval']:
                    burrow = burrow_track.last
                    burrow_color = 'red' if burrow.refined else 'orange'
                    debug_video.add_polygon(burrow.outline, burrow_color,
                                            is_closed=True, mark_points=True)
                    debug_video.add_polygon(burrow.get_centerline(self.ground),
                                            burrow_color, is_closed=False,
                                            width=2, mark_points=True)
        
            # indicate the mouse position
            if len(self.tracks) > 0:
                for obj in self.tracks:
                    if self.result['mouse/moved_first_in_frame'] is None:
                        obj_color = 'r'
                    elif obj.is_moving():
                        obj_color = 'w'
                    else:
                        obj_color = 'b'
                    debug_video.add_polygon(obj.get_track(), '0.5', is_closed=False)
                    debug_video.add_circle(obj.last_pos, self.params['mouse/model_radius'], obj_color, thickness=1)
                
            else: # there are no current tracks
                for mouse_pos in self._mouse_pos_estimate:
                    debug_video.add_circle(mouse_pos, self.params['mouse/model_radius'], 'k', thickness=1)
            
            # add additional debug information
            debug_video.add_text(str(self.frame_id), (20, 20), anchor='top')   
            debug_video.add_text(self.debug['video.mark.text1'], (300, 20), anchor='top')
            debug_video.add_text(self.debug['video.mark.text2'], (300, 50), anchor='top')
            if self.debug.get('video.mark.rect1'):
                debug_video.add_rectangle(self.debug['rect1'])
            if self.debug.get('video.mark.points'):
                debug_video.add_points(self.debug['video.mark.points'], radius=4, color='y')
            if self.debug.get('video.mark.highlight', False):
                debug_video.add_rectangle((0, 0, self.video.size[0], self.video.size[1]), 'w', 10)
                self.debug['video.mark.highlight'] = False


    def debug_finalize(self):
        """ close the video streams when done iterating """
        if 'video' in self.debug:
            try:
                self.debug['video'].close()
            except IOError:
                self.logger.exception('Error while writing out the debug video') 
            
    