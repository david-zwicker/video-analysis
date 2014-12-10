# Output statistics

## Global statistics
These are statistics that are only produced for the entire video

* `burrow_area_total`: The total area of all burrow structures
* `burrow_length_total`: The total length of all burrow structures  
* `burrow_length_max`: The length of the longest burrow

All burrow statistics are currently obtained by sweeping the mouse trail over time.
The mouse trail is given by the connecting line of the current mouse position
with the ground line.
The statistics are calculated until the end of the analysis period, which is
typically the end of the night. 

## Local statistics
These are statistics can be calculated for any time period of the video
The time slice analyzed is characterized by the following values:

* `frame_bins`: The number of the first and last frame taken into consideration
* `period_start`: The beginning of the analysis period in seconds 
* `period_end`: The end of the analysis period in seconds 
* `period_duration`: The duration of the analysis period in seconds 

For each period, the following statistics can be calculated:

* `ground_removed`: The size of the area which was below the ground line in the
first frame and is now above ground in the last frame  
* `ground_accrued`: The size of the area which was above the ground line in the
first frame and is now below ground in the last frame
* `time_spent_moving`: The total time the mouse spent moving around during the
analysis period. The mouse is said to be moving if its speed is above a threshold
value.
* `time_spent_digging`: The total time the mouse spent digging. Here, digging is
defined by the mouse being close to the end of a burrow, independent of its other
states. Consequently, this statistics reports the mouse as digging even if it is
just sitting at the end of the burrow. 
* `mouse_speed_mean`: The mean speed of the mouse during the analysis period.
Here, we assume a speed of zero for periods where we could not detect the mouse.
* `mouse_speed_mean_valid`: The mean speed of the mouse during the analysis period.
The difference to the value `mouse_speed_mean` is that here only time periods are
included in the analysis, in which we could 
actually detect the mouse.
* `mouse_speed_max`: The maximal speed the mouse attained during the analysis
period.
* `mouse_distance`: The total distance the mouse covered over the analysis period.
* `mouse_trail_longest`: The longest distance the mouse has been under ground
during the analysis period. This distance is given by the maximum over the length
of all mouse trails.

