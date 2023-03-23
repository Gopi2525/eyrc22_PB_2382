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

yellow=[]
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(640, 480))
stream = camera.capture_continuous(rawCapture,format="bgr", use_video_port=True)

for frame in stream:
    image = frame.array
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([20, 100, 100])
    upper = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    if (np.sum(mask==255) !=0) :
        yellow.append(np.sum(mask==255))
        print(yellow)
    cv2.imshow("image",image)
    cv2.imshow("mask",mask)
    key = cv2.waitKey(1) & 0xFF
       
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
        
        # if the `q` key was pressed, break from the loop and close display window
    if key == ord("q"):
        cv2.destroyAllWindows()
        break
