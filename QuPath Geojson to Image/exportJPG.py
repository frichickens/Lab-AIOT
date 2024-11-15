# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:49:30 2024

@author: bao
"""

import json
import numpy as np
from PIL import Image
from PIL import ImageDraw

# rate
r = 4
w, h = int(121863/r),int(195745/r)

# ratio aspect of MRXS and Label
w_ratio, h_ratio = 121863/w, 195745/h

data = np.zeros((h, w, 3), dtype=np.uint8)
img = Image.new('RGB', (w, h), 'white') 

# Json file location
input_file = r"Export\final.geojson"

# Load Json file
with open(input_file) as f:
    j_data = json.load(f)

for each in range(len(j_data)-1,-1,-1):
    type_name = (j_data[each]['geometry']['type'])
    
    if type_name == 'MultiPolygon':
        for layer in range(len(j_data[each]['geometry']['coordinates'])):
            for shape in range(len(j_data[each]['geometry']['coordinates'][layer])):
                cords = np.array(j_data[each]['geometry']['coordinates'][layer][shape])
                x, y = cords.T
                x = (x/w_ratio).astype(int)
                y = (y/h_ratio).astype(int)
                
                label_color = tuple(j_data[each]['properties']['classification']['color'])
                draw = ImageDraw.Draw(img)
                draw.polygon(list(zip(x,y)), label_color, None)    
                if shape < 1:
                    draw.polygon(list(zip(x,y)), label_color, None)
                else:
                    if (img.getpixel((x[0],y[0])) == label_color):
                        draw.polygon(list(zip(x,y)), (255,255,255), None)
                    else:
                        draw.polygon(list(zip(x,y)), label_color, None)
                
    else:
        shapes = len(j_data[each]['geometry']['coordinates'])
        for shape in range(shapes):
            cords = np.array(j_data[each]['geometry']['coordinates'][shape])
            x, y = cords.T
            x = (x/w_ratio).astype(int)
            y = (y/h_ratio).astype(int)
            label_color = tuple(j_data[each]['properties']['classification']['color'])
            draw = ImageDraw.Draw(img)
            if shape < 1:
                draw.polygon(list(zip(x,y)), label_color, None)
            else:
                if (img.getpixel((x[0],y[0])) == label_color):
                    draw.polygon(list(zip(x,y)), (255,255,255), None)
                else:
                    draw.polygon(list(zip(x,y)), label_color, None)

import cv2

img = np.array(img)
downS = img.copy() 
downS = cv2.pyrDown(downS) 

# cv2.imwrite('OpenCV.jpg',img)
cv2.imwrite('Normal.jpg',img)
cv2.imwrite('DownSize.jpg',downS)