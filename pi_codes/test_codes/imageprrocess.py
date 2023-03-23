import numpy as np
import cv2
import time
import RPi.GPIO as GPIO
import sys
import datetime
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
from threading import Thread
from picamera.array import PiRGBArray
from picamera import PiCamera

def clustering(image):
    
    img = image.copy()
    
    twoDimage = img.reshape((-1,3))
    twoDimage = np.float32(twoDimage)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    attempts=0
    ret,label,center=cv2.kmeans(twoDimage,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    
    return result_image
    

def remove_reflective_pixels(image):
    
    img = image.copy()
    
    blr = cv2.medianBlur(img, 15)
    # now grab brightness V of HSV here - but Gray is possibly as good
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    val = hsv[:, :, 2]
    # use ADAPTIVE_THRESH_GAUSSIAN to find spots. 
    # I manually tweaked the values- these seem to work well with what I have.
    at = cv2.adaptiveThreshold(np.array(255 - val), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 17)
    # Now invert the threshold, and run another for edges.
    ia = np.array(255 - at)  # inversion of adaptiveThreshold of the value.
    iv = cv2.adaptiveThreshold(ia, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 9)
    # ib = merged edges with the dots (as an invert mask).
    ib = cv2.subtract(iv, ia)
    # Turn this to a 3 channel mask.
    bz = cv2.merge([ib, ib, ib])
    # Use the blur where the mask is, otherwise use the image.
    dsy = np.where(bz == (0, 0, 0), blr, img)
    result = dsy
    
    return result


if __name__ == "__main__":
    
    ##################	ADD YOUR CODE HERE	##################
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 30
    rawCapture = PiRGBArray(camera, size=(640, 480))
    stream = camera.capture_continuous(rawCapture,format="bgr", use_video_port=True)
    
    for frame in stream:
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        image = frame.array
        
        ref_pix_removed = remove_reflective_pixels(image)
        ref_pix_removed = remove_reflective_pixels(ref_pix_removed)
        clustered = clustering(ref_pix_removed)
        
        cv2.imshow("Image",image)
        #cv2.imshow("Ref_pix_removed",ref_pix_removed)
        cv2.imshow("Clustered",clustered)
        
        # show the frame
        key = cv2.waitKey(1) & 0xFF
        
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
        
        # if the `q` key was pressed, break from the loop and close display window
        if key == ord("q"):
            cv2.destroyAllWindows()
            break
    