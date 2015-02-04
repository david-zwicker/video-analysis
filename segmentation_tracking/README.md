This package contains a tracker for locating mouse tails in microscope videos.
The algorithm locates tails in a given video and follows them over time.
It also locates the center line of the tail and does line scans on both sides,
which can then be plotted as kymographs for further analysis.


TODO
----
* do something about touching tails
* store line scans as kymographs beside tails
* align the line scans of different times using the patterns found
    - add GUI for manual alignment (left click moves row to left, right click to right)
* introduce logging for easy debugging