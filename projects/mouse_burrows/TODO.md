* Use the GrabCut algorithm to refine predugs
* Write a script for locating the predugs in videos without rerunning the analysis
* Write a manual of assumption we make about the videos in order to track everything
* Gather more statistics about why certain parts of the algorithm failed
* Refine the centerline by moving the points more toward the actual center
	- this could be done by segmenting the current centerline in equally spaced segments
	- each inner point on this center line can be put at the midpoint between the outlines
		to prevent problems, the point should be displaced by at most 0.5*burrow_width
	- Alternatively, find centerline by using active snake guided by current centerline and distance map
* Pass1:
* Pass2:
* Pass3:
    - detect when burrow has multiple exits and adjust the centerline accordingly
* Pass4:
* Analysis:
    - time_burrow_grew is overestimated, since we also count the instances at
        which the ground line is moved
    - time_burrowing = mouse in burrow and burrow extends some time later (at
        the same position?)
* Fix detection status of ffmpeg-errors that we recovered from
    => When checking for ffmpeg-errors, check also whether they are followed by
    an "FFmpeg error occurred! Repeat the analysis." and do not issue a warning
    in that case  
* Debug multi-threading or turn even turn it off for some applications
    => currently, multi-threading fails on odyssey and is thus disabled there


Performance improvements:
-------------------------
* Make sure that images and masks are not copied to often (rather use internal cache structures, which should be faster)
	- Do operations in place as often as possible
* Generally cache all kernels for morphological operations, since these are costly to make


Low priority enhancements:
--------------------------
