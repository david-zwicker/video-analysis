This package contains a tracker for locating mouse tails in microscope videos.
The algorithm locates tails in a given video and follows them over time.
It also locates the center line of the tail and does line scans on both sides,
which can then be plotted as kymographs for further analysis.


TODO
----
* do something about touching tails
* align the line scans of different times using the patterns found
    - add GUI for manual alignment (left click moves row to left, right click to right)
* think about using negative line tension
* report times in hours/minutes