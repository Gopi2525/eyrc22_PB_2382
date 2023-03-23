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
import math

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

pi_pwm_right = GPIO.PWM(enb,1250)
pi_pwm_left = GPIO.PWM(ena,1250)

pi_pwm_right.start(0)        
pi_pwm_left.start(0)

class pid:

    def __init__(self):

        self.p=0
        self.i=0
        self.d=0
        self.signal=0
        self.error=0
        self.error_past=0
        self.sp=0
        self.integral=0
        self.derivative=0

        self.lsp = 0
        self.rsp = 0
        self.l_base_pwm = 0
        self.r_base_pwm = 0
        self.pwm_min = 30
        self.pwm_max = 70

    def setPID(self,p,i,d):

        self.p=p
        self.i=i
        self.d=d

    def setSetPoint(self,sp):

        self.sp=math.radians(sp)
        
    def set_min_max_pwm(self,min_val,max_val):
        
        self.pwm_min = min_val
        self.pwm_max = max_val

    def base_pwm(self,l,r):

        self.l_base_pwm = l
        self.r_base_pwm = r

    def pwm_output(self,sensor):

        sensor = math.radians(sensor)

        self.error=self.sp-sensor
        self.integral = self.integral + (self.error*0.01)
        self.derivative = (self.error - self.error_past) / 0.01
        self.signal = (self.p*self.error + self.i*self.integral + self.d*self.derivative)
        self.error_past=self.error

        self.lsp = self.l_base_pwm + self.signal
        self.rsp = self.r_base_pwm - self.signal
        
        if self.lsp > self.pwm_max :
            self.lsp = self.pwm_max
            
        if self.rsp > self.pwm_max :
            self.rsp = self.pwm_max
            
        if self.lsp < self.pwm_min :
            self.lsp = self.pwm_min
            
        if self.rsp < self.pwm_min :
            self.rsp = self.pwm_min
            
        return self.lsp,self.rsp

class storage:
        
    def yellow_detectors():
        storage.yellow_count = []
        
def set_wheel_pwm(left_pwm,right_pwm):
    
    GPIO.output(in1,GPIO.HIGH)
    GPIO.output(in2,GPIO.LOW)
    GPIO.output(in3,GPIO.LOW)
    GPIO.output(in4,GPIO.HIGH)
    
    print("left : ",left_pwm,"right :",right_pwm)
    
    pi_pwm_left.ChangeDutyCycle(left_pwm)
    pi_pwm_right.ChangeDutyCycle(right_pwm)

def lane_detection_and_command_generator(image):
    
    img = image.copy()
    img2 = image.copy()
    
    ### clustering ###
    
    twoDimage = img.reshape((-1,3))
    twoDimage = np.float32(twoDimage)
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
    #cv2.imshow("thresh", thresh)

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
    
    return img2,theta

def node_detection(image):
    
    img = image.copy()
    
    ### contrast enhancement
    xp = [0, 64, 128, 192, 255]
    fp = [0, 16, 128, 240, 255]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')
    img_1 = cv2.LUT(img, table)
    
    ### yellow_deection
    img_1 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow_lower = np.array([15,100,0])
    yellow_upper = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(img_1, yellow_lower, yellow_upper)
    yellow_output = cv2.bitwise_and(image, image, mask=mask_yellow)
    yellow_ratio =(cv2.countNonZero(mask_yellow))/(image.size/3)
    yellow_ratio = np.round(yellow_ratio*100, 2)
    
    #cv2.imshow("yellow region",yellow_output)
    #print(yellow_ratio)
    
    storage.yellow_count.append(yellow_ratio)
    
def node_checker():
    
    n = 0
    print("going on")
    
    if len(storage.yellow_count) >= 8 and storage.yellow_count[-2] > 1.5 and storage.yellow_count[-1] == 0 :
        
        n = 1
        print(storage.yellow_count)
        storage.yellow_count.clear()
        print("reached node")
        
    return n

def extreme_turns(command):
    """
            
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.HIGH)
    GPIO.output(in3,GPIO.HIGH)
    GPIO.output(in4,GPIO.LOW)
    
    duty_left = 45
    duty_right = 45
    
    pi_pwm_right.ChangeDutyCycle(duty_right)
    pi_pwm_left.ChangeDutyCycle(duty_left)
    
    time.sleep(0.01)
    
    pi_pwm_right.ChangeDutyCycle(0)
    pi_pwm_left.ChangeDutyCycle(0)
    """
    
    if command == 6:
        
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
        
        duty_left = 40
        duty_right = 40
        
        pi_pwm_right.ChangeDutyCycle(duty_right)
        pi_pwm_left.ChangeDutyCycle(duty_left)
        
        pressed = 0
        i = 0
        
        while 1:
                        
            if not GPIO.input(encoder_right):
                    if not pressed:
                        
                        if i == pulses_count+5:
                            
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

def move_bot(theta):
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
    
    if theta > 15 :
        
        print("align left")
        
        GPIO.output(in1,GPIO.LOW)
        GPIO.output(in2,GPIO.HIGH)
        GPIO.output(in3,GPIO.LOW)
        GPIO.output(in4,GPIO.HIGH)
        
        duty_left = 45
        duty_right = 45
        pi_pwm_right.ChangeDutyCycle(duty_right)
        pi_pwm_left.ChangeDutyCycle(duty_left)
        
        time.sleep(0.05)
        
        duty_left = 0
        duty_right = 0
        pi_pwm_right.ChangeDutyCycle(duty_right)
        pi_pwm_left.ChangeDutyCycle(duty_left)
        
    if theta < -15 :
        
        print("align right")
        
        GPIO.output(in1,GPIO.HIGH)
        GPIO.output(in2,GPIO.LOW)
        GPIO.output(in3,GPIO.HIGH)
        GPIO.output(in4,GPIO.LOW)
        
        duty_left = 45
        duty_right = 45
        pi_pwm_right.ChangeDutyCycle(duty_right)
        pi_pwm_left.ChangeDutyCycle(duty_left)
        
        time.sleep(0.1)
        
        duty_left = 0
        duty_right = 0
        pi_pwm_right.ChangeDutyCycle(duty_right)
        pi_pwm_left.ChangeDutyCycle(duty_left)
        
    
        

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
    
    storage.yellow_detectors()
    
    global bot
    
    bot = pid()
    bot.setPID(4.0,0.5,0.01)
    bot.setSetPoint(0)
    bot.base_pwm(35,25)
    bot.set_min_max_pwm(0,50)
    
    for frame in stream:
        
        image = frame.array
        img = image.copy()
        
        navi_img,theta = lane_detection_and_command_generator(img)
        node_detection(image)
        n = node_checker()
        
        if n == 1 :
            
            extreme_turns(6)
        
        if theta <= 15 and theta >= -15 :         
            left_pwm,right_pwm = bot.pwm_output(theta)
            #print("left : ",left_pwm,"right :",right_pwm)
            set_wheel_pwm(left_pwm,right_pwm)
            
        else :
            
            move_bot(theta)
        
        #cv2.imshow("Frame", img)
        cv2.imshow("Navigation", navi_img)
        
        key = cv2.waitKey(1) & 0xFF
        rawCapture.truncate(0)
               
        if key == ord("q"):
            cv2.destroyAllWindows()
            
            pi_pwm_left.ChangeDutyCycle(0)
            pi_pwm_right.ChangeDutyCycle(0)
            
            break


    ##########################################################

    pass
