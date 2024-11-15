import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely import wkt

# Load the image
image = cv2.imread('label.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


edged = cv2.Canny(gray, threshold1=100, threshold2=200)

# Find contours
contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

print(len(contours))

# Create a list to hold polygons
polygons = []

# Iterate over each contour and convert it to a Shapely Polygon
for contour in contours:
    if len(contour) > 2 :  # Ensure the contour has enough points to be a polygon
        contour_points = [(point[0][0], point[0][1]) for point in contour]
        polygon = Polygon(contour_points)
        if polygon.is_valid:
            polygons.append(polygon)

# Create a MultiPolygon if there are multiple polygons
multi_polygon = MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]

# Get the boundary of the MultiPolygon
boundary = multi_polygon.boundary

# Convert the boundary to WKT format
boundary_wkt = boundary.wkt

cv2.drawContours(image, contours, -1, (255,141,161), 1) 
  
cv2.imwrite('Contours.jpg', image) 
# Print the WKT of the boundary
print("Boundary in WKT format:")
print(len(boundary_wkt))

#open and read the file after the appending:
f = open("output.txt", "w")
# f.write(boundary_wkt)
f.write(boundary_wkt)
f.close()