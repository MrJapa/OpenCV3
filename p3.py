import os
import cv2 as cv   
import numpy as np

os.chdir()

haystack_img = cv.imread('albion_farm.jpg', cv.IMREAD_UNCHANGED)
needle_img = cv.imread('cabbage.jpg', cv.IMREAD_UNCHANGED)

needle_w = needle_img.shape[1]
needle_h = needle_img.shape[0]

result = cv.matchTemplate(haystack_img, needle_img, cv.TM_CCOEFF_NORMED)

threshold = 0.37
locations = np.where(result >= threshold)
locations = list(zip(*locations[::-1]))
#print(locations)

rectangles = []
for loc in locations:
    rect = [int(loc[0]), int(loc[1]), needle_w, needle_h]
    rectangles.append(rect)
    rectangles.append(rect)
print(rectangles)

rectangles, weights = cv.groupRectangles(rectangles, 1, 0.5)
print(rectangles)

if len(rectangles):
    print("Found cabbage.")
    
    line_color = (0,255,0)
    line_type = cv.LINE_4
    marker_color = (255,0,255)
    marker_type = cv.MARKER_CROSS

    for (x, y, w, h) in rectangles:
        
        top_left = (x,y)
        bottom_right = (x + w, y + h)
        cv.rectangle(haystack_img,top_left,bottom_right,line_color,line_type)
        
        '''
        center_x = x + int(w/2)
        center_y = y + int(h/2) 
        cv.drawMarker(haystack_img, (center_x, center_y), marker_color, marker_type)
        '''
    cv.imshow('matches',haystack_img)
    cv.waitKey()
else:
    print("Cabbage not found")

