'''
Created on Oct 2, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

Module that contains the class responsible for the fourth pass of the algorithm
'''

from __future__ import division

import functools
import time

import cv2
import numpy as np
from scipy import cluster
from shapely import geometry

from .objects.burrow import Burrow, BurrowTrackList
from .data_handler import DataHandler
from .utils import unique_based_on_id
from video.analysis import image, curves, regions
from video.io import ImageWindow, VideoFile
from video.filters import FilterMonochrome
from video.utils import display_progress
from video.composer import VideoComposer

import debug  # @UnusedImport
from mousetracking.algorithm.utils import NormalDistribution


class FourthPass(DataHandler):
    """ class containing methods for the third pass, which locates burrows
    based on the mouse movement """
    
    def __init__(self, name='', parameters=None, **kwargs):
        super(FourthPass, self).__init__(name, parameters, **kwargs)
        if kwargs.get('initialize_parameters', True):
            self.log_event('Pass 3 - Initialized the third pass analysis.')
        self.initialize_pass()
        

    @classmethod
    def from_third_pass(cls, third_pass):
        """ create the object directly from the second pass """
        # create the data and copy the data from first_pass
        obj = cls(third_pass.name, initialize_parameters=False)
        obj.data = third_pass.data
        obj.params = obj.data['parameters']
        obj.result = obj.data.create_child('pass4')

        # close logging handlers and other files        
        third_pass.close()
        
        # initialize parameters
        obj.initialize_parameters()
        obj.initialize_pass()
        obj.log_event('Pass 3 - Initialized the third pass analysis.')
        return obj
    
    
    def initialize_pass(self):
        """ initialize values necessary for this run """
        self.params = self.data['parameters']
        self.result = self.data.create_child('pass4')
        self.result['code_status'] = self.get_code_status()
        self.debug = {}
        if self.params['debug/output'] is None:
            self.debug_output = []
        else:
            self.debug_output = self.params['debug/output']
        self._cache = {}
            

    def process(self):
        """ processes the entire video """

        self.log_event('Pass 4 - Started initializing the video analysis.')

        self.setup_processing()
        self.debug_setup()

        self.log_event('Pass 4 - Started iterating through the video with '
                       '%d frames.' % self.video.frame_count)
        self.data['analysis-status'] = 'Initialized video analysis'
        start_time = time.time()            
        
        try:
            # skip the first frame, since it has already been analyzed
            self._iterate_over_video(self.video)
                
        except (KeyboardInterrupt, SystemExit):
            # abort the video analysis
            self.video.abort_iteration()
            self.log_event('Pass 4 - Analysis run has been interrupted.')
            self.data['analysis-status'] = 'Partly finished third pass'
            
        else:
            # finalize all active burrow tracks
            if self.params['burrows/enabled_pass4']:
                self.add_burrows_to_tracks(self.active_burrows())
            
            self.log_event('Pass 4 - Finished iterating through the frames.')
            self.data['analysis-status'] = 'Finished third pass'
            
        finally:
            # cleanup in all cases 
            self.add_processing_statistics(time.time() - start_time)        
                        
            # cleanup and write out of data
            self.video.close()
            self.debug_finalize()
            self.write_data()

            
    def add_processing_statistics(self, time):
        """ add some extra statistics to the results """
        frames_analyzed = self.frame_id + 1
        self.data['pass4/video/frames_analyzed'] = frames_analyzed
        self.result['statistics/processing_time'] = time
        self.result['statistics/processing_fps'] = frames_analyzed/time


    def setup_processing(self):
        """ sets up the processing of the video by initializing caches etc """
        # load the video
        video_extension = self.params['output/video/extension']
        filename = self.get_filename('background' + video_extension, 'debug')
        self.video = FilterMonochrome(VideoFile(filename))
        
        # initialize data structures
        self.frame_id = -1

        if self.params['burrows/enabled_pass4']:
            self.result['burrows/tracks'] = BurrowTrackList()
            self.burrow_mask = np.zeros(self.video.shape[1:], np.uint8)

        
    def _iterate_over_video(self, video):
        """ internal function doing the heavy lifting by iterating over the video """
        
        # load data from previous passes
        ground_profile = self.data['pass2/ground_profile']

        # iterate over the video and analyze it
        for background_id, frame in enumerate(display_progress(self.video)):
            # calculate frame id in the original video
            self.frame_id = background_id * self.params['output/video/period'] 
            
            # copy frame to debug video
            if 'video' in self.debug:
                self.debug['video'].set_frame(frame, copy=False)
            
            # retrieve data for current frame
            self.ground = ground_profile.get_ground_profile(self.frame_id)

            # find the changes in the background
            if self.params['burrows/enabled_pass4']:
                self.find_burrows(frame)

            # store some debug information
            self.debug_process_frame(frame)
            
            if background_id % 1000 == 0:
                self.logger.debug('Analyzed frame %d', self.frame_id)

    
    #===========================================================================
    # DETERMINE THE BURROW CENTERLINES
    #===========================================================================


    def _get_burrow_exits(self, outline):
        """ determines the exits of a burrow.
        Returns a list of exits, where each exit is described by a list of
        points lying on the burrow outline
        """
        
        ground_line = self.ground.linestring
        dist_max = self.params['burrows/ground_point_distance']
        #/2
#         dist_max = dist + self.params['burrows/width']
        
        outline = curves.make_curve_equidistant(outline, spacing=2)
        
        # determine burrow points close to the ground
        exit_points = [point for point in outline
                       if ground_line.distance(geometry.Point(point)) < dist_max]

        if len(exit_points) < 2:
            return exit_points
        
        exit_points = np.array(exit_points)

        # cluster the points to detect multiple connections 
        # this is important when a burrow has multiple exits to the ground
        dist_max = self.params['burrows/width']
        data = cluster.hierarchy.fclusterdata(exit_points, dist_max,
                                              method='single', 
                                              criterion='distance')
        
        exits = [exit_points[data == cluster_id]
                 for cluster_id in np.unique(data)]
        
#         exits = []
#         for cluster_id in np.unique(data):
#             points = exit_points[data == cluster_id]
#             point = points.mean(axis=0)
#             point_ground = curves.get_projection_point(ground_line, point)
#             exits.append(point_ground)
            
        return exits
    
    
    def _get_burrow_centerline(self, burrow, points_start, points_end=None):
        """ determine the centerline of a burrow with one exit """
            
        ground_line = self.ground.linestring
            
        # get a binary image of the burrow
        mask, shift = burrow.get_mask(margin=2, dtype=np.int32, ret_shift=True)
        
        # mark the start points according to their distance to the ground line
#         dists_g = [ground_line.distance(geometry.Point(p))
#                    for p in points_start]
        points_start = curves.translate_points(points_start, -shift[0], -shift[1])
        for p in points_start:
            mask[p[1], p[0]] = 1

        if points_end is None:
            # end point is not given and will thus be determined automatically

            # calculate the distance from the start point 
            regions.distance_fill(mask.T, points_start)
            
            
            # find the second point by locating the farthest point
            _, _, _, p_end = cv2.minMaxLoc(mask)
        
        else:
            # prepare the end point if present

            # translate that point to the mask frame
            points_end = curves.translate_points(points_end, -shift[0], -shift[1])
            for p in points_end:
                mask[p[1], p[0]] = 1

            # calculate the distance from the start point 
            regions.distance_fill(mask.T, points_start, points_end)
            
            # get the distance between the start and the end point
            dists = [mask[p[1], p[0]] for p in points_end]
            best_endpoint = np.argmin(dists)
            p_end = points_end[best_endpoint]
            
        # find an estimate for the centerline from the shortest distance from
        # the end point to the burrow exit
        points = regions.shortest_path_in_distance_map(mask, p_end)

#         debug.show_shape(geometry.MultiPoint(points_start),
#                          geometry.Point(p_end),
#                          background=mask)
#         exit()

        # translate the points back to global coordinates 
        centerline = curves.translate_points(points, shift[0], shift[1])
        # save centerline such that burrow exit is first point
        centerline = centerline[::-1]
        
        # add points that might be outside of the burrow outline
        ground_start = curves.get_projection_point(ground_line, centerline[0]) 
        centerline.insert(0, ground_start)
        if points_end is not None:
            ground_end = curves.get_projection_point(ground_line, centerline[-1]) 
            centerline.append(ground_end)
            
        # simplify the curve        
        centerline = cv2.approxPolyDP(np.array(centerline, np.int),
                                      epsilon=1, closed=False)
            
        # save the centerline in the burrow structure
        burrow.centerline = centerline[:, 0, :]
            

    def determine_burrow_centerline(self, burrow):
        """ determines the burrow centerlines """
        exits = self._get_burrow_exits(burrow.outline)
        
        # check the different number of possible exits
        if len(exits) == 0:
            self.logger.debug('%d: Found burrow with no exit at %s',
                              self.frame_id, burrow.position)
            return
        elif len(exits) == 1:
            self._get_burrow_centerline(burrow, exits[0])
        elif len(exits) == 2:
            self._get_burrow_centerline(burrow, exits[0], exits[1])
        else:
            self.logger.warn('%d: Found burrow with more than 2 exits at %s',
                             self.frame_id, burrow.position)
            return
    
    
    #===========================================================================
    # LOCATE BURROWS
    #===========================================================================


    def get_ground_mask(self, y_displacement=0, margin=0, fill_value=255):
        """ returns a binary mask distinguishing the ground from the sky """
        # build a mask with potential burrows
        width, height = self.video.size
        mask_ground = np.zeros((height, width), np.uint8)
        
        # create a mask for the region below the current mask_ground profile
        ground_points = np.empty((len(self.ground) + 4, 2), np.int32)
        ground_points[:-4, :] = self.ground.points
        ground_points[:, 1] += y_displacement
        
        # extend the outline points to the edges of the mask
        ground_points[-4, :] = (width - margin, ground_points[-5, 1])
        ground_points[-3, :] = (width - margin, height - margin)
        ground_points[-2, :] = (margin, height - margin)
        ground_points[-1, :] = (margin, ground_points[0, 1])
        
        # create the mask
        cv2.fillPoly(mask_ground, np.array([ground_points], np.int32),
                     color=fill_value)

        return mask_ground


    def _get_image_statistics(self, img, mask, prior=128, kernel='box'):
        """ calculate mean and variance in a window around a point, 
        excluding the point itself
        prior denotes a value that is subtracted from the frame before
            calculating statistics. This is necessary for numerical stability.
            The prior should be close to the mean of the values.
        """
        # calculate the window size
        window = int(self.params['burrows/image_statistics_window'])
        ksize = 2*window + 1
        
        # check for possible integer overflow (very conservatively)
        if np.iinfo(np.int).max < (ksize*max(prior, 255 - prior))**2:
            raise RuntimeError('Window is too large and an integer overflow '
                               'could happen.')

        # prepare the function that does the actual filtering
        if kernel == 'box':
            filter_image = functools.partial(cv2.boxFilter, ddepth=-1,
                                             ksize=(ksize, ksize), normalize=False,
                                             borderType=cv2.BORDER_CONSTANT)
        
        elif kernel == 'ellipse' or kernel == 'circle':        
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               ksize=(ksize, ksize))
            filter_image = functools.partial(cv2.filter2D, ddepth=-1,
                                             kernel=kernel,
                                             borderType=cv2.BORDER_CONSTANT)
            
        else:
            raise ValueError('Unknown filter kernel `%s`' % kernel)

        # create the image from which the statistics will be calculated             
        img_m = np.zeros_like(img, np.int)
        img_m[mask] = img[mask].astype(np.int) - prior
        
        # calculate how many on pixel there are in each region
        mask = np.asarray(mask, np.int)
        count = filter_image(mask)
        count = count - mask #< exclude the central point if it was in the mask
        count[count < 2] = 2 #< limit the minimal count to avoid division by zero
        
        # calculate the local sums and sums of squares
        img_m2 = img_m**2
        s1 = filter_image(img_m)
        s2 = filter_image(img_m2)
        s1 = s1 - mask*img_m  #< exclude the central point
        s2 = s2 - mask*img_m2 #< exclude the central point
        # don't use -= here, since s1 seems to be int32 only 
        
        # calculate mean and variance
        mean = s1/count + prior
        var = (s2 - s1**2/count)/(count - 1)

        # return the associated normal distributions
        return NormalDistribution(mean, var, count)


    def update_burrow_mask(self, frame):
        """
        updates the burrow mask based on the current frame.
        """
        # get the mask of the current ground
        ground_mask = self.get_ground_mask(fill_value=1)
        # add the sky region to the burrow mask (since it is also background)
        self.burrow_mask[ground_mask == 0] = 1

        # define the region where the frame of the cage is
        left, right = self.ground.points[0, 0], self.ground.points[-1, 0]
        def disable_frame_region(img):
            """ helper function setting the region of the cage frame to False """
            img[:, :left] = False
            img[:, right:] = False
            return img
        
        # get masks for the region of the sand and the background
        mask_sand = disable_frame_region(self.burrow_mask == 0)
        mask_back = disable_frame_region(self.burrow_mask == 1)

        # get statistics of these two regions
        stats_sand = self._get_image_statistics(frame, mask_sand)
        stats_back = self._get_image_statistics(frame, mask_back)

        # build the mask of the region we consider
        count_min = 0.02*stats_back.count.max()
        mask = (stats_sand.count > count_min) & (stats_back.count > count_min)  
        mask[ground_mask == 0] = False
        mask = disable_frame_region(mask)
        
#         debug.show_image(stats_sand.mean, np.sqrt(stats_sand.var),
#                          stats_back.mean, np.sqrt(stats_back.var),
#                          mask=mask)
#         exit()

        # restrict the mask to points where the distributions differ significantly
        overlap = stats_sand.overlap(stats_back)
        mask[mask] = (overlap[mask] < 0.5)
        
        # determine the probabilities 
        prob_sand = stats_sand.pdf(frame[mask], mask)
        prob_back = stats_back.pdf(frame[mask], mask)

        # determine points in the mask that belong to burrows
        burrow_points = (prob_back > prob_sand)
        self.burrow_mask[mask] = burrow_points

        # remove chunks close to the ground line 
        w = 2*int(self.params['burrows/width']/2) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(w, w))
        mask = cv2.erode(ground_mask, kernel)
        self.burrow_mask[ground_mask - mask == 1] = 0 
               
        # connect burrow chunks
        self.burrow_mask = cv2.morphologyEx(self.burrow_mask, cv2.MORPH_CLOSE, kernel)
        
        # there are no burrows above the ground by definition
        self.burrow_mask[ground_mask == 0] = 0
        
#         debug.show_image(frame, self.burrow_mask)
#         exit()        

#         # Label all features to remove the small, bright ones
#         labels, num_features = ndimage.measurements.label(self.burrow_mask)
#
#         # calculate the overlap between the probability distributions
#         overlap = stats_sand.overlap(stats_back)
#  
#         for label in xrange(1, num_features + 1):
#             mask = (labels == label)
#             # remove burrows with too little overlap
#             # we can't really tell burrow and sand apart
#             # it's safer to assume there is no burrow
#             if np.mean(overlap[mask]) > 0.5:
#                 self.burrow_mask[mask] = 0


    def get_burrow_chunks(self, frame):
        """ determines regions under ground that belong to burrows """

#         # remove too small burrows
#         ksize = 2*int(self.params['burrows/width_min']) + 1
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
#         mask = cv2.morphologyEx(self.burrow_mask, cv2.MORPH_OPEN, kernel)
# 
#         # connect burrow chunks        
#         ksize = 2*int(self.params['burrows/chunk_dist_max']) + 1
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
         
        #labels, num_features = ndimage.measurements.label(self.burrow_mask)
        # extend the contour to the ground line
        contours, _ = cv2.findContours(self.burrow_mask,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        self.debug['video.mark.text1'] = '#chunks: %d' % len(contours)

        burrow_chunks = []
        for contour in contours:
            if len(contour) <= 2:
                continue

            # check whether the burrow chunk is large enough
            props = image.regionprops(contour=contour)
            if props.area < self.params['burrows/chunk_area_min']:
                continue

            # remove problematic parts of the outline
            polygon = geometry.Polygon(np.array(contour[:, 0, :], np.double))
            polygon_buffered = polygon.buffer(0)
            if isinstance(polygon_buffered, geometry.Polygon):
                polygon_buffered = [polygon_buffered]
                
            #contour = regions.regularize_contour_points(contour[:, 0, :])
            
            # save the contour line as a burrow
            for polygon in polygon_buffered:
                if len(polygon.exterior.coords) > 2:
                    burrow_chunks.append(polygon.exterior.coords)
                
        return burrow_chunks


    def _connect_burrow_to_structure(self, contour, structure):
        """ extends the burrow outline such that it connects to the ground line 
        or to other burrows """

        outline = geometry.Polygon(contour)

        # determine burrow points close to the structure
        dist = structure.distance(outline)
        conn_points = []
        while len(conn_points) == 0:
            dist += self.params['burrows/width']/2
            conn_points = [point for point in contour
                           if structure.distance(geometry.Point(point)) < dist]
        
        conn_points = np.array(conn_points)

        # cluster the points to detect multiple connections 
        # this is important when a burrow has multiple exits to the ground
        if len(conn_points) >= 2:
            dist_max = self.params['burrows/width']
            data = cluster.hierarchy.fclusterdata(conn_points, dist_max,
                                                  method='single', 
                                                  criterion='distance')
        else:
            data = np.ones(1, np.int)
            
        burrow_width_min = self.params['burrows/width_min']
        for cluster_id in np.unique(data):
            p_exit = conn_points[data == cluster_id].mean(axis=0)
            p_ground = curves.get_projection_point(structure, p_exit)
            
            line = geometry.LineString((p_exit, p_ground))
            tunnel = line.buffer(distance=burrow_width_min/2,
                                 cap_style=geometry.CAP_STYLE.flat)

            # add this to the burrow outline
            outline = outline.union(tunnel.buffer(0.1))
        
        # get the outline points
        outline = regions.get_enclosing_outline(outline)
        outline = regions.regularize_linear_ring(outline)
        outline = np.array(outline.coords)
        
        # fill the burrow mask, such that this extension does not have to be
        # done next time again
        cv2.fillPoly(self.burrow_mask, [np.asarray(outline, np.int32)], 1) 

        return outline
        
        
    def connect_burrow_chunks(self, burrow_chunks):
        """ takes a list of burrow chunks and connects them such that in the
        end all burrow chunks are connected to the ground line. """
        if len(burrow_chunks) == 0:
            return []
        
        dist_max = self.params['burrows/chunk_dist_max']

        # build the contour profiles of the burrow chunks        
        linear_rings = [geometry.LinearRing(c) for c in burrow_chunks]
        
        # calculate distances to ground
        dist_func = self.ground.linestring.distance
        ground_dist = [dist_func(ring) for ring in linear_rings]
            
        # calculate distances to other burrows
        burrow_dist = np.empty([len(burrow_chunks)]*2)
        np.fill_diagonal(burrow_dist, np.inf)
        for x, contour1 in enumerate(linear_rings):
            for y, contour2 in enumerate(linear_rings[x+1:], x+1):
                dist = contour1.distance(contour2)
                burrow_dist[x, y] = dist
                burrow_dist[y, x] = dist
        
        # handle all burrows close to the ground
        connected, disconnected = [], []
        for k in xrange(len(burrow_chunks)):
            if ground_dist[k] < np.min(burrow_dist[k]):
                # burrow is closer to ground than to any other burrow
                if 1 < ground_dist[k] < dist_max:
                    burrow_chunks[k] = \
                        self._connect_burrow_to_structure(burrow_chunks[k],
                                                          self.ground.linestring)
                connected.append(k)
            else:
                disconnected.append(k)
                
        assert (set(connected) | set(disconnected)) == set(range(len(burrow_chunks)))
        
        # handle all remaining chunks, which need to be connected to other chunks
        while connected and disconnected:
            # find chunks which is closest to all the others
            dist = burrow_dist[disconnected, :][:, connected]
            k1, k2 = np.unravel_index(dist.argmin(), dist.shape)
            if dist[k1, k2] > dist_max:
                # don't connect structures that are too far from each other
                break
            c1, c2 = disconnected[k1], connected[k2]
            # k1 is chunk to connect, k2 is closest chunk to connect it to

            # connect the current chunk to the other structure
            structure = geometry.LinearRing(burrow_chunks[c2])
            enlarged_chunk = self._connect_burrow_to_structure(burrow_chunks[c1], structure)
            
            # merge the two structures
            poly1 = geometry.Polygon(enlarged_chunk)
            poly2 = regions.regularize_polygon(geometry.Polygon(structure))
            poly = poly1.union(poly2).buffer(0.1)
            
            # find and regularize the common outline
            outline = regions.get_enclosing_outline(poly)
            outline = regions.regularize_linear_ring(outline)
            outline = outline.coords
            
            # replace the current chunk by the merged one
            burrow_chunks[c1] = outline
            
            # replace all other burrow chunks with the same id
            id_c2 = id(burrow_chunks[c2])
            for k, bc in enumerate(burrow_chunks):
                if id(bc) == id_c2:
                    burrow_chunks[k] = outline
            
            # mark the cluster as connected
            del disconnected[k1]
            connected.append(c1)

        # return the unique burrow structures
        burrows = []
        connected_chunks = (burrow_chunks[k] for k in connected) 
        for outline in unique_based_on_id(connected_chunks):
            outline = regions.regularize_contour_points(outline)
            try:
                burrow = Burrow(outline)
            except ValueError:
                continue
            else:
                if burrow.area >= self.params['burrows/area_min']:
                    burrows.append(burrow)
        
        return burrows


    def active_burrows(self):
        """ returns a generator to iterate over all active burrows """
        for burrow_track in self.result['burrows/tracks']:
            if burrow_track.active:
                yield burrow_track.last


    def add_burrows_to_tracks(self, burrows):
        """ adds the burrows to the current tracks """
        burrow_tracks = self.result['burrows/tracks']
        
        # get currently active tracks
        active_tracks = [track for track in burrow_tracks
                         if track.active]

        # check each burrow that has been found
        tracks_extended = set()
        for burrow in burrows:
            for track_id, track in enumerate(active_tracks):
                if burrow.intersects(track.last):
                    # burrow belongs to a known track => add it
                    tracks_extended.add(track_id)
                    if burrow != track.last:
                        self.determine_burrow_centerline(burrow)
                        track.append(self.frame_id, burrow)
                    break
            else:
                # burrow is not known => start a new track
                self.determine_burrow_centerline(burrow)
                burrow_tracks.create_track(self.frame_id, burrow)
                
        # deactivate tracks that have not been found
        for track_id, track in enumerate(active_tracks):
            if track_id not in tracks_extended:
                track.active = False


    def find_burrows(self, frame):
        """ finds burrows from the current frame """
        # find regions of possible burrows            
        self.update_burrow_mask(frame)

        # identify chunks from the burrow mask
        burrow_chunks = self.get_burrow_chunks(frame)
        
        self.debug['video.mark.text1'] += ', %d' % len(burrow_chunks)

        # get the burrows by connecting chunks
        burrows = self.connect_burrow_chunks(burrow_chunks)
        
        # assign the burrows to burrow tracks or create new ones
        # this also determines their centerlines
        self.add_burrows_to_tracks(burrows)


    #===========================================================================
    # DEBUGGING
    #===========================================================================


    def debug_setup(self):
        """ prepares everything for the debug output """
        # load parameters for video output        
        video_output_period = 1#int(self.params['output/video/period'])
        video_extension = self.params['output/video/extension']
        video_codec = self.params['output/video/codec']
        video_bitrate = self.params['output/video/bitrate']
        
        # set up the general video output, if requested
        if 'video' in self.debug_output or 'video.show' in self.debug_output:
            # initialize the writer for the debug video
            debug_file = self.get_filename('pass4' + video_extension, 'debug')
            self.debug['video'] = VideoComposer(debug_file, size=self.video.size,
                                                fps=self.video.fps, is_color=True,
                                                output_period=video_output_period,
                                                codec=video_codec,
                                                bitrate=video_bitrate)
            
            if 'video.show' in self.debug_output:
                name = self.name if self.name else ''
                multiprocessing = self.params['debug/use_multiprocessing']
                position = self.params['debug/window_position']
                image_window = ImageWindow(self.debug['video'].shape,
                                           title='Debug video pass 4 [%s]' % name,
                                           multiprocessing=multiprocessing,
                                           position=position)
                self.debug['video.show'] = image_window


    def debug_process_frame(self, frame):
        """ adds information of the current frame to the debug output """
        
        if 'video' in self.debug:
            debug_video = self.debug['video']
            
            # plot the ground profile
            if self.ground is not None:
                debug_video.add_line(self.ground.points, is_closed=False,
                                     mark_points=True, color='y')
                
            if self.params['burrows/enabled_pass4']:
                debug_video.highlight_mask(self.burrow_mask == 1, 'b', strength=128)
                for burrow in self.active_burrows():
                    debug_video.add_line(burrow.outline, 'r')
                    if burrow.centerline is not None:
                        debug_video.add_line(burrow.centerline, 'r',
                                             is_closed=False, width=2,
                                             mark_points=True)
                        
                # additional values
                debug_video.add_text(self.debug.get('video.mark.text1', ''),
                                     (300, 20), anchor='top')
                debug_video.add_text(self.debug.get('video.mark.text2', ''),
                                     (300, 50), anchor='top')
                
            # add additional debug information
            if 'video.show' in self.debug:
                if debug_video.output_this_frame:
                    self.debug['video.show'].show(debug_video.frame)
                else:
                    self.debug['video.show'].show()


    def debug_finalize(self):
        """ close the video streams when done iterating """
        # close the window displaying the video
        if 'video.show' in self.debug:
            self.debug['video.show'].close()
        
        # close the open video streams
        if 'video' in self.debug:
            try:
                self.debug['video'].close()
            except IOError:
                    self.logger.exception('Error while writing out the debug '
                                          'video') 
            
    