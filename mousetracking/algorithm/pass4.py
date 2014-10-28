'''
Created on Oct 2, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

Module that contains the class responsible for the third pass of the algorithm
'''

from __future__ import division

import time

import cv2
import numpy as np
from scipy import cluster, ndimage
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

        # do several runs with increasing thresholds until we succeed
        self.result['burrows/detection_threshold'] = 0
        pass_finished = False
        try:
            while not pass_finished:
                self.result['burrows/detection_threshold'] += 1
                pass_finished = self.process_run()
        except (KeyboardInterrupt, SystemExit):
            pass
        
        # write out results
        self.write_data()


    def process_run(self):
        self.setup_processing()
        self.debug_setup()

        self.log_event('Pass 4 - Started iterating through video with %d '
                       'frames at threshold %d.'
                       % (self.video.frame_count,
                          self.result['burrows/detection_threshold']))
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
            # finished analysis successfully or aborted
            if not self.abort_iteration:
                # finalize all active burrow tracks
                self.add_burrows_to_tracks(self.active_burrows())
                
                self.log_event('Pass 4 - Finished iterating through the frames.')
                self.data['analysis-status'] = 'Finished third pass'
            
        finally:
            # cleanup in all cases 
            self.add_processing_statistics(time.time() - start_time)        
                        
            # cleanup and write out of data
            self.video.close()
            self.debug_finalize()

        if self.abort_iteration:
            # pass has not been finished
            return False
        else:
            return True            

            
    def add_processing_statistics(self, time):
        """ add some extra statistics to the results """
        frames_analyzed = self.frame_id + 1
        self.data['pass4/video/frames_analyzed'] = frames_analyzed
        self.result['statistics/processing_time'] = time
        self.result['statistics/processing_fps'] = frames_analyzed/time


    def setup_processing(self):
        """ sets up the processing of the video by initializing caches etc """
        # load the video
        #cropping_rect = self.data['pass1/video/cropping_rect'] 
        #video_info = self.load_video(cropping_rect=cropping_rect)
        
        video_extension = self.params['output/video/extension']
        filename = self.get_filename('background' + video_extension, 'debug')
        self.video = FilterMonochrome(VideoFile(filename))
        
        # initialize data structures
        self.frame_id = -1
        self.abort_iteration = False
        self.result['burrows/mask_overflow'] = 0
        self.background_avg = None

        self.result['burrows/tracks'] = BurrowTrackList()
        self.burrow_mask = None
        self._cache['image_uint8'] = np.empty(self.video.shape[1:], np.uint8)

        
    def _iterate_over_video(self, video):
        """ internal function doing the heavy lifting by iterating over the video """
        
        # load data from previous passes
        ground_profile = self.data['pass2/ground_profile']
        adaptation_rate = self.params['background/adaptation_rate']

        # iterate over the video and analyze it
        for background_id, frame in enumerate(display_progress(self.video)):
            self.frame_id = background_id * self.params['output/video/period'] 
            
            # adapt the background to current frame
            if self.background_avg is None:
                self.background_avg = frame.astype(np.double)
            else:
                self.background_avg += adaptation_rate*(frame - self.background_avg)
            
            # copy frame to debug video
            if 'video' in self.debug:
                self.debug['video'].set_frame(frame, copy=False)
            
            # retrieve data for current frame
            self.ground = ground_profile.get_ground_profile(self.frame_id)

            # find the changes in the background
            if background_id >= 0*1/adaptation_rate:
                self.find_burrows(frame)

            if self.abort_iteration:
                break

            # store some debug information
            self.debug_process_frame(frame)
            
            if self.frame_id % 100000 == 0:
                self.logger.debug('Analyzed frame %d', self.frame_id)

    
    #===========================================================================
    # DETERMINE THE BURROW CENTERLINES
    #===========================================================================


    def _get_burrow_exits(self, outline):
        """ determines the exits of a burrow """
        
        ground_line = self.ground.linestring
        dist = self.params['burrows/ground_point_distance']
        dist_max = dist + self.params['burrows/width']
        
        outline = curves.make_curve_equidistant(outline, spacing=dist)
        
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
        
        exits = []
        for cluster_id in np.unique(data):
            points = exit_points[data == cluster_id]
            point = points.mean(axis=0)
            point_ground = curves.get_projection_point(ground_line, point)
            exits.append(point_ground)
            
        return exits
    
    
    def _get_burrow_centerline(self, burrow, point_start, point_end=None):
        """ determine the centerline of a burrow with one exit """

        def get_closest_point(point):
            """ get point closest to the burrow outline """
            rel_points = burrow.outline - np.asarray(point)
            contour = np.r_[rel_points, [rel_points[0]]]
    
            # In principle, we could use         
            #     return curves.get_projection_point(contour, (0, 0))
            # but this is too slow.
            # We thus test many different points on the outline and just
            # take the closest
            contour = curves.make_curve_equidistant(contour, 2)
            k = np.argmin(np.linalg.norm(contour, axis=1))
            return contour[k] + np.asarray(point)
            
        # find the point in the burrow that is closest to the ground point
        p_start = get_closest_point(point_start)

        # get a binary image of the burrow
        mask, shift = burrow.get_mask(margin=3, dtype=np.int32, ret_shift=True)
        p_start = (int(p_start[0] - shift[0]),
                   int(p_start[1] - shift[1]))
        mask[p_start[1], p_start[0]] = 1
        
        if point_end is None:
            # end point is not given and will thus be determined automatically
            
            # calculate the distance from the start point 
            regions.distance_fill(mask.T, [p_start])
            
            # find the second point by locating the farthest point
            _, dist, _, p_end = cv2.minMaxLoc(mask)
        
        else:
            # prepare the end point if present
            p_end = get_closest_point(point_end)
            # translate that point to the mask frame
            p_end = (int(p_end[0] - shift[0]),
                     int(p_end[1] - shift[1]))
            mask[p_end[1], p_end[0]] = 1

            # calculate the distance from the start point 
            regions.distance_fill(mask.T, [p_start], [p_end])
            
            # get the distance between the start and the end point
            dist = mask[p_end[1], p_end[0]]

        # find an estimate for the centerline from the shortest distance between
        # the two points p_start and p_end
        points = []
        xmax = mask.shape[1] - 1
        ymax = mask.shape[0] - 1
        x, y = p_end
        d = int(dist)
        while d > 1:
            points.append((x, y))
            d -= 1
            
            # test diagonal steps
            if x > 0 and y > 0 and mask[y - 1, x - 1] == d - 1:
                x -= 1
                y -= 1
                d -= 1
            elif x > 0 and y < ymax and mask[y + 1, x - 1] == d - 1:
                x -= 1
                y += 1
                d -= 1
            elif x < xmax and y > 0 and mask[y - 1, x + 1] == d - 1:
                x += 1
                y -= 1
                d -= 1
            elif x < xmax and y < ymax and mask[y + 1, x + 1] == d - 1:
                x += 1
                y += 1
                d -= 1 
            
            # test horizontal and vertical steps
            elif x > 0 and mask[y, x - 1] == d:
                x -= 1
            elif x < xmax and mask[y, x + 1] == d:
                x += 1
            elif y > 0 and mask[y - 1, x] == d:
                y -= 1
            elif y < ymax and mask[y + 1, x] == d:
                y += 1
        points.append(p_start)
                
#         debug.show_shape(geometry.LineString(points),
#                          geometry.Point(p_start), geometry.Point(p_exit_b),
#                          background=mask)

        # translate the points back to global coordinates 
        centerline = curves.translate_points(points, shift[0], shift[1])
        # save centerline such that burrow exit is first point
        centerline = centerline[::-1]
        
        # add points that might be outside of the burrow outline 
        centerline.insert(0, point_start)
        if point_end is not None:
            centerline.append(point_end)
            
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
    # LOCATE BURROWS FROM CHANGES IN THE BACKGROUND
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
        
        cv2.fillPoly(mask_ground, np.array([ground_points], np.int32),
                     color=fill_value)

        return mask_ground


    def get_initial_burrow_mask(self, frame):
        """ get the burrow mask estimated from the first frame.
        This is mainly the predug, provided in the antfarm experiments """
        ground_mask = self.get_ground_mask()
        w = int(self.params['burrows/ground_point_distance'])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w, w))
        ground_mask = cv2.erode(ground_mask, kernel)
        
        # find initial threshold color
        color_sand = self.data['pass1/colors/sand']
        color_sky = self.data['pass1/colors/sky']
        fraction = 0.33
        color_thresh = fraction*color_sand + (1 - fraction)*color_sky
        
        largest_area = np.inf
        while largest_area > 10*self.params['burrows/area_min']:
            # apply the threshold
            self.burrow_mask = (frame < color_thresh).astype(np.uint8)
            self.burrow_mask[ground_mask == 0] = 0
            
            # find the largest cluster
            labels, num_features = ndimage.measurements.label(self.burrow_mask)
            areas = ndimage.measurements.sum(labels, labels, index=range(1, num_features + 1))
            largest_area = max(areas)
            
            # change fraction in case we have to do another round
            color_thresh -= 1
            
        # remove all structures which are too far away from the ground line
        labels, num_features = ndimage.measurements.label(self.burrow_mask)
        for label in xrange(1, num_features + 1):
            # check whether the burrow is large enough
            props = image.regionprops(labels == label)
            
            # check whether the burrow is sufficiently underground
            ground_line = self.ground.linestring
            dist = ground_line.distance(geometry.Point(props.centroid))
            if dist > self.params['burrows/ground_point_distance']:
                self.burrow_mask[label == labels] = 0
                    

    def update_burrow_mask_old(self, frame):
        """ determines a mask of all the burrows """
        # initialize masks for this frame
        ground_margin = self.params['burrows/ground_point_distance']/2
        ground_mask = self.get_ground_mask(y_displacement=ground_margin,
                                           fill_value=1)
        mask = self._cache['image_uint8']
        
        # determine the difference between this frame and the background
        diff = -self.background_avg + frame

        change_threshold = self.data['pass1/colors/sand_std']
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
#         # shrink burrows
#         mask[:] = (diff > change_threshold)
#         #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel1)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
#          
#         self.burrow_mask[mask == 1] = 0

        # enlarge burrows with excavated regions
        mask[:] = (diff < -change_threshold)
        if mask.sum() > 0.1*ground_mask.sum():
            # there is way too much change
            # This can happen when the light flickers
            # => we ignore these frames and return immediately
            return
        
        # remove small changes, which are likely due to noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
        # add the regions to the burrow mask
        self.burrow_mask[mask == 1] = 1
        # combine nearby burrow chunks by closing the mask
        self.burrow_mask = cv2.morphologyEx(self.burrow_mask, cv2.MORPH_CLOSE, kernel1)
        # ensure that all points above ground are not burrow chunks
        self.burrow_mask[ground_mask == 0] = 0


    def update_burrow_mask(self, frame):
        """ determines a mask of all the burrows """
        # initialize masks for this frame
        ground_margin = self.params['burrows/ground_point_distance']
        ground_mask = self.get_ground_mask(y_displacement=ground_margin,
                                           #margin=50,
                                           fill_value=1)
        #mask = self._cache['image_uint8']
        
        burrow_width = int(self.params['burrows/width'])
        
        #frame = frame.copy()
        color = frame[ground_mask == 1].mean()
        frame[ground_mask == 0] = color
        
        # adaptive thresholding
        mask = cv2.adaptiveThreshold(frame, 1,
                                     cv2.THRESH_BINARY,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     2*burrow_width + 1,
                                     C=self.result['burrows/detection_threshold'])
        
#         # shrink burrows
#         mask[:] = (diff > change_threshold)
#         #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel1)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
#          
#         self.burrow_mask[mask == 1] = 0

        # enlarge burrows with excavated regions
        if mask.sum() > 0.15*ground_mask.sum():
            # The mask covers too much area!
            # There are two cases in which this can happen:
            # 1) the light flickers => we ignore these frames and return
            # 2) wrong threshold => restart iteration with new threshold
            # We distinguish these two cases by counting how the mask
            # covers too much area. In case 1, we expect this to be rarely
            # the case, while in the case 2 almost every frame should
            # be too bright
            self.result['burrows/mask_overflow'] += 1
            if self.result['burrows/mask_overflow'] > 5:
                # completely abort 
                self.abort_iteration = True
            return
        else:
            self.result['burrows/mask_overflow'] = 0
        
        
        # remove small changes, which are likely due to noise
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (burrow_width//2, burrow_width//2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
        # add the regions to the burrow mask
        #self.burrow_mask[mask == 1] = 1
        # combine nearby burrow chunks by closing the mask
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*burrow_width, 2*burrow_width))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel1)
        
        self.burrow_mask = mask
        
        # ensure that all points above ground are not burrow chunks
        #qself.burrow_mask[ground_mask == 0] = 0
        #debug.show_image(self.burrow_mask, frame)


    def get_burrow_chunks(self, frame):
        """ determines regions under ground that belong to burrows """     
        ground_line = self.ground.linestring

        color_sand = self.data['pass1/colors/sand']
        color_sky = self.data['pass1/colors/sky']
        fraction = self.params['burrows/color_threshold_fraction']
        color_thresh = fraction*color_sand + (1 - fraction)*color_sky

        labels, num_features = ndimage.measurements.label(self.burrow_mask)

        burrow_chunks = []
        for label in xrange(1, num_features + 1):
            # check whether the burrow is large enough
            props = image.regionprops(labels == label)
            if props.area < self.params['burrows/area_min']:
                continue
            
            # check whether the burrow is sufficiently underground
            dist = ground_line.distance(geometry.Point(props.centroid))
            if dist < self.params['burrows/ground_point_distance']/2:
                continue
            
            # check mean color of the burrow
            color_avg = frame[labels == label].mean()
            if color_avg > color_thresh:
                self.burrow_mask[labels == label] = 0
                continue
            
            # extend the contour to the ground line
            contours, _ = cv2.findContours(np.asarray(labels == label, np.uint8),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 1:
                # a single contour has been found
                contour = np.squeeze(np.asarray(contours, np.double))
            elif len(contours) == 0:
                # no contour has been found
                continue
            else:
                # multiple contours have been found
                # => return the one with the largest area
                contour_max, area_max = None, 0
                for contour in contours:
                    contour = np.squeeze(np.asarray(contour, np.double))
                    props = image.regionprops(contour=contour)
                    if props.area > area_max:
                        area_max = props.area
                        contour_max = contour
                contour = contour_max
            
            contour = regions.regularize_contour_points(contour)
            
            # save the contour line as a burrow
            if len(contour) > 2:
                burrow_chunks.append(contour)
                
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
        
        # fill the burrow mask, such that this extension does not have to be done next time again
        cv2.fillPoly(self.burrow_mask, [np.asarray(outline, np.int32)], 1) 

        return outline
        
        
    def connect_burrow_chunks(self, burrow_chunks):
        """ takes a list of burrow chunks and connects them such that in the
        end all burrow chunks are connected to the ground line. """
        if len(burrow_chunks) == 0:
            return []
        
        dist_max = self.params['burrows/chunk_dist_max']
        
        # calculate distances to ground
        ground_dist = []
        for contour in burrow_chunks:
            # measure distance to ground
            outline = geometry.LinearRing(contour)
            dist = self.ground.linestring.distance(outline)
            ground_dist.append(dist)
            
        # calculate distances to other burrows
        burrow_dist = np.zeros([len(burrow_chunks)]*2)
        np.fill_diagonal(burrow_dist, np.inf)
        linear_rings = [geometry.LinearRing(c) for c in burrow_chunks]
        for x, contour1 in enumerate(linear_rings):
            for y, contour2 in enumerate(linear_rings[x+1:], x+1):
                dist = contour1.distance(contour2)
                burrow_dist[x, y] = dist
                burrow_dist[y, x] = dist
        
        # handle all burrows close to the ground
        connected = []
        disconnected = []
        for k in xrange(len(burrow_chunks)):
            if ground_dist[k] < np.min(burrow_dist[k]):
                # burrow is closest to ground
                if 1 < ground_dist[k] < dist_max:
                    burrow_chunks[k] = \
                        self._connect_burrow_to_structure(burrow_chunks[k],
                                                          self.ground.linestring)
                connected.append(k)
            else:
                disconnected.append(k)
                
        # make sure that at least one chunk is connected to the ground
#         if len(connected) == 0:
#             # find the structure closest to the ground
#             k = np.argmin(ground_dist)
#             burrow_chunks[k] = \
#                 self._connect_burrow_to_structure(burrow_chunks[k],
#                                                   self.ground.linestring)
#             connected.append(k)
#             disconnected.remove(k)
                
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
        if self.burrow_mask is None:
            self.get_initial_burrow_mask(frame)
        
        # find regions of possible burrows            
        self.update_burrow_mask(frame)

        # identify chunks from the burrow mask
        burrow_chunks = self.get_burrow_chunks(frame)

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
                                                codec=video_codec, bitrate=video_bitrate)
            
            if 'video.show' in self.debug_output:
                name = self.name if self.name else ''
                position = self.params['debug/window_position']
                image_window = ImageWindow(self.debug['video'].shape,
                                           title='Debug video pass 4 [%s]' % name,
                                           multiprocessing=self.params['debug/use_multiprocessing'],
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
                
            debug_video.highlight_mask(self.burrow_mask == 1, 'b', strength=128)
            for burrow in self.active_burrows():
                debug_video.add_line(burrow.outline, 'r')
                if burrow.centerline is not None:
                    debug_video.add_line(burrow.centerline, 'r', is_closed=False,
                                         width=2, mark_points=True)
                
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
            
    