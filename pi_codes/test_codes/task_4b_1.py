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

in1=32
in2=33
ena=31
in3=38
in4=40
enb=37

encoder_right = 24
encoder_left = 26

GPIO.setup(in1,GPIO.OUT)
GPIO.setup(in2,GPIO.OUT)
GPIO.setup(in3,GPIO.OUT)
GPIO.setup(in4,GPIO.OUT)
GPIO.setup(ena,GPIO.OUT)
GPIO.setup(enb,GPIO.OUT)

GPIO.setup(encoder_right, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(encoder_left, GPIO.IN, pull_up_down=GPIO.PUD_UP)

GPIO.output(in1,0)
GPIO.output(in2,0)
GPIO.output(in3,0)
GPIO.output(in4,0)
GPIO.output(ena,0)
GPIO.output(enb,0)

pi_pwm_right = GPIO.PWM(enb,1000)
pi_pwm_left = GPIO.PWM(ena,1000)

pi_pwm_right.start(0)        
pi_pwm_left.start(0)

##############################################################

def control_logic(img,img2):

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
    
    ### exposure correction ###
    contrast = 5. # Contrast control ( 0 to 127)
    brightness = 2. # Brightness control (0-100)
    img = cv2.addWeighted( img, contrast, img, 0, brightness)
    twoDimage = img.reshape((-1,3))
    twoDimage = np.float32(twoDimage)

    ### clustering ###
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    attempts=0
    ret,label,center=cv2.kmeans(twoDimage,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    #cv2.imshow('Output', result_image)

    ### centroid detection ###
    gray_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY_INV)
    M = cv2.moments(thresh)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 320, 350
    cv2.circle(img2, (cX, cY), 5, (255, 255, 255), -1)
    cv2.putText(img2, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    ### draw line and find theta ###
    mp = (320,480)
    mpup = (320,350)
    if cX == mp[0]:
        cX = cX+0.01
    m = (cY - mp[1])/(cX-mp[0])
    theta = round(np.arctan(1/m)*(180/3.14))
    cv2.circle(img2,mp, 5, (255, 255, 255), -1)
    cv2.putText(img2, str(theta), (mp[0] , mp[1] - 50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    img2 = cv2.line(img2, (int(cX), int(cY)), mp, (255, 255, 255), 2)
    img2 = cv2.line(img2, mpup, mp, (0, 255, 0), 2)

    #cv2.imshow("Final",img2)
    
    ### decission making ###
    
    if theta < 5 and theta > -5 :
        
        command = 0
        
    elif theta >= 5 and theta <= 60 :
        
        command = 2
        
    elif theta <= -5 and theta >= -60 :
        
        command = -2
        
    elif theta >= 45 :
        
        command = 4
        
    elif theta <= -45 :
        
        command = 4
        
    else:
        
        command = 101
    
    return result_image,img2,theta,command


    ##########################################################

def move_bot(command):
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
    
    if command == 0 :
    
        print("Move bot straight")
        
        t1 = datetime.datetime.now()
        
        GPIO.output(in1,GPIO.HIGH)
        GPIO.output(in2,GPIO.LOW)
        GPIO.output(in3,GPIO.LOW)
        GPIO.output(in4,GPIO.HIGH)
        
        duty_left = 73
        duty_right = 70
        
        duty_left = 45
        duty_right = 43
        
        pi_pwm_right.ChangeDutyCycle(duty_right)
        pi_pwm_left.ChangeDutyCycle(duty_left)
        
        """
            
        while(1):

            if (datetime.datetime.now()- t1).total_seconds() >= 2:
                            
                pi_pwm_right.ChangeDutyCycle(0)
                pi_pwm_left.ChangeDutyCycle(0)
                            
                return
                
        """
            
    elif command == 1:
        
        print("Turn bot left ")
        
        b = 13.1         #wheel_base
        r = 6.3/2        #wheel_radius
        theta = 0.226893 #13degree
        s = r*theta
        pulses_count  = int((3.14*b)/(8*s))
        
        GPIO.output(in1,GPIO.LOW)
        GPIO.output(in2,GPIO.HIGH)
        GPIO.output(in3,GPIO.LOW)
        GPIO.output(in4,GPIO.HIGH)
        
        duty_left = 80
        duty_right = 80
        
        pi_pwm_right.ChangeDutyCycle(duty_right)
        pi_pwm_left.ChangeDutyCycle(duty_left)
        
        pressed = 0
        i = 0
        
        while 1:
                        
            if not GPIO.input(encoder_right):
                    if not pressed:
                        
                        if i == pulses_count:
                            
                            duty_left = 0
                            duty_right = 0
                            pi_pwm_right.ChangeDutyCycle(duty_right)
                            pi_pwm_left.ChangeDutyCycle(duty_left)
                            
                            break
                                                
                        i = i+1
                        pressed = 1
                                                    
            else:
                pressed = 0
            
            time.sleep(0.001)
            
    elif command == 4:
        
        print("Turn bot left ")
        
        b = 13.1         #wheel_base
        r = 6.3/2        #wheel_radius
        theta = 0.226893 #13degree
        s = r*theta
        pulses_count  = int((3.14*b)/(16*s))
        
        GPIO.output(in1,GPIO.LOW)
        GPIO.output(in2,GPIO.HIGH)
        GPIO.output(in3,GPIO.LOW)
        GPIO.output(in4,GPIO.HIGH)
        
        duty_left = 80
        duty_right = 80
        
        pi_pwm_right.ChangeDutyCycle(duty_right)
        pi_pwm_left.ChangeDutyCycle(duty_left)
        
        pressed = 0
        i = 0
        
        while 1:
                        
            if not GPIO.input(encoder_right):
                    if not pressed:
                        
                        if i == pulses_count:
                            
                            duty_left = 0
                            duty_right = 0
                            pi_pwm_right.ChangeDutyCycle(duty_right)
                            pi_pwm_left.ChangeDutyCycle(duty_left)
                            
                            break
                                                
                        i = i+1
                        pressed = 1
                                                    
            else:
                pressed = 0
            
            time.sleep(0.001)    

            
    elif command == -1:
        
        print("Turn bot right ")
        
        b = 13.1         #wheel_base
        r = 6.3/2        #wheel_radius
        theta = 0.226893 #13degree
        s = r*theta
        pulses_count  = int((3.14*b)/(8*s))
        
        GPIO.output(in1,GPIO.HIGH)
        GPIO.output(in2,GPIO.LOW)
        GPIO.output(in3,GPIO.HIGH)
        GPIO.output(in4,GPIO.LOW)
        
        duty_left = 85
        duty_right = 85
        
        pi_pwm_right.ChangeDutyCycle(duty_right)
        pi_pwm_left.ChangeDutyCycle(duty_left)
        
        pressed = 0
        i = 0
        
        while 1:
                        
            if not GPIO.input(encoder_right):
                    if not pressed:
                        
                        if i == pulses_count:
                            
                            duty_left = 0
                            duty_right = 0
                            pi_pwm_right.ChangeDutyCycle(duty_right)
                            pi_pwm_left.ChangeDutyCycle(duty_left)
                            
                            break
                                                
                        i = i+1
                        pressed = 1
                                                    
            else:
                pressed = 0
            
            time.sleep(0.001)
            
    elif command == -4:
        
        print("Turn bot right ")
        
        b = 13.1         #wheel_base
        r = 6.3/2        #wheel_radius
        theta = 0.226893 #13degree
        s = r*theta
        pulses_count  = int((3.14*b)/(16*s))
        
        GPIO.output(in1,GPIO.HIGH)
        GPIO.output(in2,GPIO.LOW)
        GPIO.output(in3,GPIO.HIGH)
        GPIO.output(in4,GPIO.LOW)
        
        duty_left = 85
        duty_right = 85
        
        pi_pwm_right.ChangeDutyCycle(duty_right)
        pi_pwm_left.ChangeDutyCycle(duty_left)
        
        pressed = 0
        i = 0
        
        while 1:
                        
            if not GPIO.input(encoder_right):
                    if not pressed:
                        
                        if i == pulses_count:
                            
                            duty_left = 0
                            duty_right = 0
                            pi_pwm_right.ChangeDutyCycle(duty_right)
                            pi_pwm_left.ChangeDutyCycle(duty_left)
                            
                            break
                                                
                        i = i+1
                        pressed = 1
                                                    
            else:
                pressed = 0
            
            time.sleep(0.001)
            
            
    elif command == 3 :
        
        time.sleep(5)
        
    elif command == -2:
        
        print("Tune bot right ")

        
        GPIO.output(in1,GPIO.HIGH)
        GPIO.output(in2,GPIO.LOW)
        GPIO.output(in3,GPIO.HIGH)
        GPIO.output(in4,GPIO.LOW)
        
        duty_left = 25
        duty_right = 25
        
        pi_pwm_right.ChangeDutyCycle(duty_right)
        pi_pwm_left.ChangeDutyCycle(duty_left)
    
        
    elif command == 2:
        
        print("Tune bot left ")
        
        GPIO.output(in1,GPIO.LOW)
        GPIO.output(in2,GPIO.HIGH)
        GPIO.output(in3,GPIO.LOW)
        GPIO.output(in4,GPIO.HIGH)
        
        duty_left = 25
        duty_right = 25
        
        pi_pwm_right.ChangeDutyCycle(duty_right)
        pi_pwm_left.ChangeDutyCycle(duty_left)    
        
    elif command == 101:
        
        duty_left = 0
        duty_right = 0
        pi_pwm_right.ChangeDutyCycle(duty_right)
        pi_pwm_left.ChangeDutyCycle(duty_left)
        
        print("Nothing to do")
        
        return


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
    
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(640, 480))
    stream = camera.capture_continuous(rawCapture,format="bgr", use_video_port=True)
    
    for frame in stream:
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        image = frame.array
        
        img = image.copy()
        img2 = image.copy()
        
        lane,result,theta,command = control_logic(img,img2)
        cv2.imshow("Lane",lane)
        cv2.imshow("Result",result)
        
        move_bot(command)
        print(theta)

        # show the frame
        key = cv2.waitKey(1) & 0xFF
        
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
        
        # if the `q` key was pressed, break from the loop and close display window
        if key == ord("q"):
            cv2.destroyAllWindows()
            break
            
            
    duty_left = 0
    duty_right = 0
    pi_pwm_right.ChangeDutyCycle(duty_right)
    pi_pwm_left.ChangeDutyCycle(duty_left)
    

    print("code executed")

    ##########################################################

    pass
