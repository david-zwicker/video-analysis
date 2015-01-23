This package contains a tracker for locating mouse tails in microscope videos.
The algorithm locates tails in a given video and follows them over time.
It also locates the center line of the tail and does line scans on both sides,
which can then be plotted as kymographs for further analysis.


TODO
----
* Try separating foreground from background by looking at the statistics in the
    neighborhood of each pixel (see http://stackoverflow.com/a/11459915/932593)
    => high standard deviation should be inside the tail (background is rather smooth)
* separate tails that touch in the initial frame
    - this could be done using watershed segmentation
* do something about touching tails
* align the line scans of different times using the patterns found
* introduce logging for easy debugging