'''
*****************************************************************************************
*
*        		===============================================
*           		Pharma Bot (PB) Theme (eYRC 2022-23)
*        		===============================================
*
*  This script is to implement Task 4B-Part 1 of Pharma Bot (PB) Theme (eYRC 2022-23).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:			[ PB 2382 ]
# Author List:		[ Aakash K , Devaprasad S , Gopi M , Ilam Thendral R  ]
# Filename:			task_4b_1.py
# Functions:		control_logic, move_bot
# 					
####################### IMPORT MODULES #######################
## You are not allowed to make any changes in this section.   ##
## You have to implement this task with the available modules ##
##############################################################

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

########### ADD YOUR UTILITY FUNCTIONS HERE ##################

def img_pre_processing(img):
    
    ##### sharpening image to avoid none type errors in hough transform 
	kernel = np.array([[0, -1, 0],[-1, 5,-1],[0, -1, 0]])
	image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
	img = image_sharp
	
	return img

##############################################################

def control_logic():

    """
    Purpose:
    ---
    This function is suppose to process the frames from the PiCamera and
    check for the error using image processing and with respect to error
    it should correct itself using PID controller.

    >> Process the Frame from PiCamera 
    >> Check for the error in line following and node detection
    >> PID controller

    Input Arguments:
    ---
    You are free to define input arguments for this function.

    Hint: frame [numpy array] from PiCamera can be passed in this function and it can
        take the action using PID 

    Returns:
    ---
    You are free to define output parameters for this function.

    Example call:
    ---
    control_logic()
    """    

    ##################	ADD YOUR CODE HERE	##################


    ##########################################################

def move_bot():
    """
    Purpose:
    ---
    This function is suppose to move the bot

    Input Arguments:
    ---
    You are free to define input arguments for this function.

    Hint: Here you can have inputs left, right, straight, reverse and many more
        based on your control_logic

    Returns:
    ---
    You are free to define output parameters for this function.

    Example call:
    ---
    move_bot()
    """    

    ##################	ADD YOUR CODE HERE	##################


    ##########################################################



################# ADD UTILITY FUNCTIONS HERE #################





##############################################################
    
if __name__ == "__main__":

    """
    The goal of the this task is to move the robot through a predefied 
    path which includes straight road traversals and taking turns at 
    nodes. 

    This script is to be run on Raspberry Pi and it will 
    do the following task.
 
    >> Stream the frames from PiCamera
    >> Process the frame, do the line following and node detection
    >> Move the bot using control logic

    The overall task should be executed here, plan accordingly. 
    """    

    ##################	ADD YOUR CODE HERE	##################
    
    task_2b = __import__('task_2b')
    
    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 15
    rawCapture = PiRGBArray(camera, size=(640, 480))
    stream = camera.capture_continuous(rawCapture,format="bgr", use_video_port=True)

    for frame in stream:
        
        image = frame.array
        img = image.copy()
        
        cv2.imwrite("sample6.jpg",img)
        cv2.imshow("Frame", img)
        key = cv2.waitKey(1) & 0xFF
        rawCapture.truncate(0)
               
        if key == ord("q"):
            cv2.destroyAllWindows()
            break


    ##########################################################

    pass
