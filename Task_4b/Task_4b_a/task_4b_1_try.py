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
        
        pi_pwm_right.ChangeDutyCycle(duty_right)
        pi_pwm_left.ChangeDutyCycle(duty_left)
            
        while(1):

            if (datetime.datetime.now()- t1).total_seconds() >= 2:
                            
                pi_pwm_right.ChangeDutyCycle(0)
                pi_pwm_left.ChangeDutyCycle(0)
                            
                return
            
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
            
    elif command == 3 :
        
        time.sleep(5)


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
            
    move_bot(0)
    time.sleep(1)

    print("code executed")

    ##########################################################

    pass
