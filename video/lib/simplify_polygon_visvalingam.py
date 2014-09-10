"""

This script is to created to simplify shapefile geometry using the
Visvalingam algorithm found here
http://www2.dcs.hull.ac.uk/CISRG/publications/DPs/DP10/DP10.html

Threshold is the area of the largest allowed triangle

This code was copied from https://github.com/ARSimmons/Shapely_Fiona_Visvalingam_Simplify
"""

__author__ = 'asimmons'


from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
import heapq
from shapely.geometry.polygon import LinearRing


class TriangleCalculator(object):
    def __init__(self, point, index):
        # Need to add better validation

        # Save instance variables
        self.point = point
        self.ringIndex = index
        self.prevTriangle = None
        self.nextTriangle = None

    # enables the instantiation of 'TriangleCalculator' to be compared
    # by the calcArea().
    def __cmp__(self, other):
        return cmp(self.calcArea(), other.calcArea())

    ## calculate the effective area of a triangle given
    ## its vertices -- using the cross product
    def calcArea(self):
        # Add validation
        if not self.prevTriangle or not self.nextTriangle:
            print "ERROR:"

        p1 = self.point
        p2 = self.prevTriangle.point
        p3 = self.nextTriangle.point
        area = abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2.0
        #print "area = " + str(area) + ", point = " + str(self.point)
        return area



def simplify_line(line, threshold):
    # unlike rings: we need to keep beginning and end points static throughout the simplification process


    # Build list of Triangles from the line points
    triangleArray = []
    ## each triangle contains an index and a point (x,y)
    # handle line 'interior' (i.e. the vertices
    #  between start and end) first -- explicitly
    # defined using the below slice notation
    # i.e. [1:-1]
    for index, point in enumerate(line.coords[1:-1]):
        triangleArray.append(TriangleCalculator(point, index))

    # then create start/end points separate from the triangleArray (meaning
    # we cannot have the start/end points included in the heap sort)
    startIndex = 0
    endIndex = len(line.coords)-1
    startTriangle = TriangleCalculator(line.coords[startIndex], startIndex)
    endTriangle = TriangleCalculator(line.coords[endIndex], endIndex)

    # Hook up triangles with next and prev references (doubly-linked list)
    # NOTE: linked list are composed of nodes, which have at
    # least one link to another node (and this is a doubly-linked list..pointing at
    # both our prevTriangle & our nextTriangle)
    # NOTE: in this code block the 'triangle' is our 'triangle node'

    for index, triangle in enumerate(triangleArray):
        # set prevIndex to be the adjacent point to index
        prevIndex = index - 1
        nextIndex = index + 1

        if prevIndex >= 0:
            triangle.prevTriangle = triangleArray[prevIndex]
        else:
            triangle.prevTriangle = startTriangle

        if nextIndex < len(triangleArray):
            triangle.nextTriangle = triangleArray[nextIndex]
        else:
            triangle.nextTriangle = endTriangle

    # Build a min-heap from the TriangleCalculator list
    # print "heapify"
    heapq.heapify(triangleArray)


    # Simplify steps...


    # Note: in contrast
    # to our function 'simplify_ring'
    # we can allow our array to go down to 0 and STILL have a valid line
    # because we will still have the start and end points
    while len(triangleArray) > 0:
        # if the smallest triangle is greater than the threshold, we can stop
        # i.e. loop to point where the heap head is >= threshold
        if triangleArray[0].calcArea() >= threshold:
            #print "break"
            break
        else:
            # print statement for debugging - prints area's and coords of deleted/simplified pts
            #print "simplify...triangle area's and their corresponding points that were less then the threshold"
            #print "area = " + str(triangleArray[0].calcArea()) + ", point = " + str(triangleArray[0].point)
            tri_prev = triangleArray[0].prevTriangle
            tri_next = triangleArray[0].nextTriangle
            tri_prev.nextTriangle = tri_next
            tri_next.prevTriangle = tri_prev
            # This has to be done after updating the linked list
            # in order for the areas to be correct when the
            # heap re-sorts
            # print "popping (i.e. re-measuring area & comparing)"
            heapq.heappop(triangleArray)
            #print "area = " + str(triangle.calcArea()) + ", point = " + str(triangle.point)
            #print "done popping (i.e. area that is less than threshold, and will have point removed)"

    # Create an list of indices from the triangleRing heap
    indexList = []
    for triangle in triangleArray:
        # add 1 b/c the triangle array's first index is actually the second point
        indexList.append(triangle.ringIndex + 1)
    # Append start and end points back into the array
    indexList.append(startTriangle.ringIndex)
    indexList.append(endTriangle.ringIndex)

    # Sort the index list
    indexList.sort()

    # Create a new simplified ring
    simpleLine = []
    for index in indexList:
        simpleLine.append(line.coords[index])

    # Convert list into LineString
    simpleLine = LineString(simpleLine)

    # print statements for debugging to check if points are being reduced...
    #print "Starting size (incl. beginning/end point): " + str(len(line.coords))
    #print "Ending size (incl. beginning/end point): " + str(len(simpleLine.coords))
    #print "Starting Coord: " + str(line.coords[startIndex])
    #print "End Coord: " + str(line.coords[endIndex])
    #print list(simpleLine.coords)
    return simpleLine


def simplify_ring(ring, threshold):

    # Build list of TriangleCalculators
    triangleRing = []
    ## each triangle contains an index and a point (x,y)
    ## because rings have a point on top of a point
    ## we are skipping the last point by using slice notation[:-1]
    ## *i.e. 'a[:-1]' # everything except the last item*
    for index, point in enumerate(ring.coords[:-1]):
        triangleRing.append(TriangleCalculator(point, index))

    # Hook up triangles with next and prev references (doubly-linked list)
    for index, triangle in enumerate(triangleRing):
        # set prevIndex to be the adjacent point to index
        # these steps are necessary for dealing with
        # closed rings
        prevIndex = index - 1
        if prevIndex < 0:
            # if prevIndex is less than 0, then it means index = 0, and
            # the prevIndex is set to last value in the index
            # (i.e. adjacent to index[0])
            prevIndex = len(triangleRing) - 1
        # set nextIndex adjacent to index
        nextIndex = index + 1
        if nextIndex == len(triangleRing):
            # if nextIndex is equivalent to the length of the array
            # set nextIndex to 0
            nextIndex = 0
        triangle.prevTriangle = triangleRing[prevIndex]
        triangle.nextTriangle = triangleRing[nextIndex]

    # Build a min-heap from the TriangleCalculator list
    heapq.heapify(triangleRing)

    # Simplify
    while len(triangleRing) > 2:
        # if the smallest triangle is greater than the threshold, we can stop
        # i.e. loop to point where the heap head is >= threshold

        if triangleRing[0].calcArea() >= threshold:
            break
        else:
            tri_prev = triangleRing[0].prevTriangle
            tri_next = triangleRing[0].nextTriangle
            tri_prev.nextTriangle = tri_next
            tri_next.prevTriangle = tri_prev
            # This has to be done after updating the linked list
            # in order for the areas to be correct when the
            # heap re-sorts
            heapq.heappop(triangleRing)

    # Handle case where we've removed too many points for the ring to be a polygon
    if len(triangleRing) < 3:
        return None

    # Create an list of indices from the triangleRing heap
    indexList = []
    for triangle in triangleRing:
        indexList.append(triangle.ringIndex)

    # Sort the index list
    indexList.sort()

    # Create a new simplified ring
    simpleRing = []
    for index in indexList:
        simpleRing.append(ring.coords[index])

    # Convert list into LinearRing
    simpleRing = LinearRing(simpleRing)

    # print statements for debugging to check if points are being reduced...
    #print "Starting size: " + str(len(ring.coords))
    #print "Ending size: " + str(len(simpleRing.coords))

    return simpleRing


def simplify_multipolygon(mpoly, threshold):
    # break multipolygon into polys
    polyList = mpoly.geoms
    simplePolyList = []

    # call simplify_polygon() on each
    for poly in polyList:
        simplePoly = simplify_polygon(poly, threshold)
        #if not none append to list
        if simplePoly:
            simplePolyList.append(simplePoly)

    # check that polygon count > 0, otherwise return None
    if not simplePolyList:
        return None

    # put back into multipolygon
    return MultiPolygon(simplePolyList)


def simplify_polygon(poly, threshold):

    # Get exterior ring
    simpleExtRing = simplify_ring(poly.exterior, threshold)

    # If the exterior ring was removed by simplification, return None
    if simpleExtRing is None:
        return None

    simpleIntRings = []
    for ring in poly.interiors:
        simpleRing = simplify_ring(ring, threshold)
        if simpleRing is not None:
            simpleIntRings.append(simpleRing)
    return Polygon(simpleExtRing, simpleIntRings)


def simplify_multiline(mline, threshold):
    # break MultiLineString into lines
    lineList = mline.geoms
    simpleLineList = []

    # call simplify_line on each
    for line in lineList:
        simpleLine = simplify_line(line, threshold)
        #if not none append to list
        if simpleLine:
            simpleLineList.append(simpleLine)

    # check that line count > 0, otherwise return None
    if not simpleLineList:
        return None

    # put back into multilinestring
    return MultiLineString(simpleLineList)

