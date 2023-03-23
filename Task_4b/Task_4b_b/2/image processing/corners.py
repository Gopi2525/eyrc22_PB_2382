import numpy as np
import cv2
"""
img = cv2.imread('road_ref.jpeg')
i = img.copy()
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

corners = cv2.goodFeaturesToTrack(gray, 1000, 0.1, 5,blockSize=50)
corners = np.int0(corners)

for corner in corners:
    x,y = corner.ravel()
    cv2.circle(img,(x,y),3,255,-1)

print(len(corners))


up = []

for corner in corners:
    x,y = corner.ravel()
    b,g,r = i[y,x]

    if b < 200 and g < 200 and r < 200 :
    	up.append(corner)

for u in up:
    x,y = u.ravel()
    cv2.circle(i,(x,y),3,255,-1)

print(len(corners))
    
cv2.imshow('Corner',img)
cv2.imshow('Corner updated',i)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

import cv2
import numpy as np

def corners(img,a):

    if a == 0 :
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    else:
        gray = img

    gray = np.float32(gray)

    corners = cv2.goodFeaturesToTrack(gray, 200, 0.1, 25)
    corners = np.int0(corners)


    for corner in corners:
        x,y = corner.ravel()
        cv2.circle(img,(x,y),3,255,-1)

    return img

def remove_glare(img):

    mask = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)

    cv2.imshow("mask",mask)
    cv2.imshow("dst",dst)
    result = cv2.bitwise_and(img,img, mask= mask)

    return dst

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('road.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        img1 = frame.copy()
        img2 = frame.copy()

        corner_img = corners(img1,0)
        corner_img_updated  = corners(remove_glare(img2),0)

        # Display the resulting frame
        #cv2.imshow('Frame',frame)
        cv2.imshow('Corners',corner_img)
        cv2.imshow('Corners_updated',corner_img_updated)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else: 
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()