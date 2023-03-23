from pylab import *
import cv2
import os, sys

img = cv2.imread("road_ref.jpeg")
image_in = img.copy()
image_in = cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB); # Load the glared image
h, s, v = cv2.split(cv2.cvtColor(image_in, cv2.COLOR_RGB2HSV)) # split into HSV components

nonSat = s > 100 # Find all pixels that are not very saturated

# Slightly decrease the area of the non-satuared pixels by a erosion operation.
disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
nonSat = cv2.erode(nonSat.astype(np.uint8), disk)
#nonSat = nonSat.astype(np.uint8)

# Set all brightness values, where the pixels are still saturated to 0.
v2 = v.copy()
v2[nonSat == 0] = 0

blurred = cv2.GaussianBlur(v2, (11, 11), 0)
thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)[1]

result = cv2.bitwise_and(img,img, mask= thresh)

"""
cv2.imshow("original",image_in)
cv2.imshow("Hue",h)
cv2.imshow("Saturation",s)
cv2.imshow("Brightness",v)
cv2.imshow("s > 180",nonSat)
cv2.imshow("Original/Brightness",v)
"""
cv2.imshow("Masked/Brightness",v2)
cv2.imshow("thresh",thresh)
cv2.imshow('Masked Image',result)

cv2.waitKey(0)
cv2.destroyAllWindows()