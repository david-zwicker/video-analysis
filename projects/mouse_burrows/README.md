This project is about tracking mice during burrowing activities. It analyzes
the dynamics of the mouse, the ground line, and the actual burrows.


Necessary python packages:

Package     | Usage                                      
------------|-------------------------------------------
cv2         | OpenCV python bindings for computer vision 
h5py        | HDF5 python binding for writing out data    
matplotlib  | Plotting library used for output           
networkx    | Graph library for graph based tracking
numpy       | Array library used for manipulating data
scipy       | Miscellaneous scientific functions
shapely     | Library for manipulating geometric shapes
yaml        | YAML binding for writing out data


Optional python packages:

Package      | Usage                                      
-------------|-------------------------------------------
dateutil     | Being less picky about date formats
descartes    | Debug plotting of shapes
faulthandler | Detecting low level crashes
grip         | Converting markdown to html 
pandas       | Writing results to csv files
pint         | Reporting results with physical units
sharedmem    | Showing the videos in a separate process while iterating 
tqdm         | Showing a progress bar while iterating
