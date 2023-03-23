import RPi.GPIO as GPIO
import time
from AlphaBot import AlphaBot

import picamera
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import numpy

Ab = AlphaBot()
Ab.stop()

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
stream = camera.capture_continuous(rawCapture,format="bgr", use_video_port=True)

for frame in stream:

    image = frame.array
    left = -50
    right = 65
    Ab.setMotor(left, right)

    cv2.imshow("Frame", image)
    rawCapture.truncate(0)

    key = cv2.waitKey(1) & 0xFF

Ab.stop()

