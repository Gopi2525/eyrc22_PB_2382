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

pi_pwm_right = GPIO.PWM(enb,1250)
pi_pwm_left = GPIO.PWM(ena,1000)

pi_pwm_right.start(0)        
pi_pwm_left.start(0)

##############################################################
class y():
    def ye():
        y.yellow=[]
        

def blackdetector(img):

    lower_bound = np.array([0, 0, 0])
    upper_bound = np.array([180,180,180])
    black = cv2.inRange(img, lower_bound, upper_bound)
    n_black_pix = np.sum(black == 255)

    return black, n_black_pix

def yellowdetector(img):

    lower_bound = np.array([50, 80, 80])
    upper_bound = np.array([80,130, 120])
    yellow = cv2.inRange(img, lower_bound, upper_bound)

    n_yellow_pix = np.sum(yellow == 255)

    return yellow, n_yellow_pix

##############################################################

def control_logic(img):
    
    ##################	ADD YOUR CODE HERE	##################
    
    img2 = img.copy()
    img4 = img.copy()
    
    ### exposure correction ###
    contrast = 5. # Contrast control ( 0 to 127)
    brightness = 2. # Brightness control (0-100)
    img = cv2.addWeighted( img, contrast, img, 0, brightness)
    twoDimage = img.reshape((-1,3))
    twoDimage = np.float32(twoDimage)
    
    img3 = img.copy()
    
    """
    ###contours###
    image_copy = img4.copy()
    Gray = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(Gray, (3, 3), 0)
    threshed = cv2.adaptiveThreshold(blurred, 255, 1, 1, 11, 2)
    contours, _ = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    c = 0 
    for i in contours:
        area = cv2.contourArea(i)
        if area > 700:
                if area > max_area:
                    max_area = area
                    best_cnt = i
                    img4 = cv2.drawContours(img4, contours, c, (0, 255, 0), 3)
        c+=1
    mask = np.zeros((Gray.shape),np.uint8)
    cv2.drawContours(mask,[best_cnt],0,255,-1)
    cv2.drawContours(mask,[best_cnt],0,0,2)
    out = np.zeros_like(Gray)
    out[mask == 255] = Gray[mask == 255]
    blurred = cv2.GaussianBlur(out, (5,5), 0)
    threshed = cv2.adaptiveThreshold(blurred, 255, 1, 1, 11, 2)
    contours, _ = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = 0
    for i in contours:
            area = cv2.contourArea(i)
            if area > 250/2:
                cv2.drawContours(img4, contours, c, (0, 255, 0), 3)
            c+=1
    cv2.imshow("Final Image", img4)
    """
    ###yellow dection###
    cy = 0
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([20, 100, 100])
    upper = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    if (np.sum(mask==255) !=0) :
        y.yellow.append(np.sum(mask==255))
        #print(sum(y.yellow),"//")
       # if(sum(y.yellow)>50):
#         cy = 1

    ###black detection###
    hsv = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)
    lower_val = np.array([0,0,0])
    upper_val = np.array([0,0,0])
    mask = cv2.inRange(hsv, lower_val, upper_val)
    res = cv2.bitwise_and(img3,img3, mask= mask)
    res2 = cv2.bitwise_not(mask)
    res2 = cv2.bitwise_not(res2)
    #cv2.imshow("img2", res2)
    a,count = blackdetector(img)
    percentage_of_black_pixels = (count/306720)*100
    if percentage_of_black_pixels != 0 :
        c = 1
    else:
        c = 0
    print(c)

    ### clustering ###
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    attempts=0
    ret,label,center=cv2.kmeans(twoDimage,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    #cv2.imshow('Output', result_image)

    ### centroid detection ###
    gray_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY_INV)
    thresh = cv2.bitwise_or(thresh,res2)
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
    

    if cy==1:
        cmd=-6
    elif theta == 0:
        cmd = 666
    elif theta <= -10 and theta >= -50 and c == 1 :
        ### align right ###
        cmd = -1
    elif theta >= 10 and theta <= 50 and c == 1:
        ### align left ###
        cmd = 1
    elif theta < 10 and theta >-10 and c == 1:
        ### move straight ###
        cmd = 0
    else:
        ### reposition me ###
        print("theta",theta,"-----","c",c)
        if theta < 0 and percentage_of_black_pixels != 0  :
            cmd = -1
        elif theta > 0 and percentage_of_black_pixels != 0:
            cmd = 1
        else:
            cmd = 666
    
    #######################################################

    return thresh,img2,theta,cmd

def move_bot(command,theta):
    
    #######################################################
    
    if command == 0 :
        print(" move straight")
        
        GPIO.output(in1,GPIO.HIGH)
        GPIO.output(in2,GPIO.LOW)
        GPIO.output(in3,GPIO.LOW)
        GPIO.output(in4,GPIO.HIGH)
        
        duty_left = 55
        duty_right = 55
        time.sleep(0.1)
        pi_pwm_right.ChangeDutyCycle(duty_right)
        pi_pwm_left.ChangeDutyCycle(duty_left)
        
        
    elif command == 1 :
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
        
    elif command  == -1:
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
        
    elif command == 7:
        
        print(" move straight yellow detected")
        
        GPIO.output(in1,GPIO.HIGH)
        GPIO.output(in2,GPIO.LOW)
        GPIO.output(in3,GPIO.LOW)
        GPIO.output(in4,GPIO.HIGH)
        
        duty_left = 50
        duty_right = 55
        
        pi_pwm_right.ChangeDutyCycle(duty_right)
        pi_pwm_left.ChangeDutyCycle(duty_left)
        
        time.sleep(0.5)
        
        duty_left = 0
        duty_right = 0
        
        pi_pwm_right.ChangeDutyCycle(duty_right)
        pi_pwm_left.ChangeDutyCycle(duty_left)
        
        return 1
        
    elif command == -6:
        
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
            
    elif command == 6:
        
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
    
    
    elif command  == 666:
        print("reposition me")
        
        GPIO.output(in1,GPIO.HIGH)
        GPIO.output(in2,GPIO.LOW)
        GPIO.output(in3,GPIO.LOW)
        GPIO.output(in4,GPIO.HIGH)
        
        duty_left = 0
        duty_right = 0
        
        pi_pwm_right.ChangeDutyCycle(duty_right)
        pi_pwm_left.ChangeDutyCycle(duty_left)
        
    else:
        print(":)")
        
    return 0
    
    #######################################################
        
if __name__ == "__main__":
    
    ##################	ADD YOUR CODE HERE	##################
    y.ye()
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 30
    rawCapture = PiRGBArray(camera, size=(640, 480))
    stream = camera.capture_continuous(rawCapture,format="bgr", use_video_port=True)
    
    start = time.time()
    till = 500
    
    
    for frame in stream:
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        image = frame.array
        
        img = image.copy()
        
                
        lane,result,theta,command = control_logic(img)
        o = move_bot(command,theta)
        cv2.imshow("Lane",lane)
        cv2.imshow("Result",result)
        
        print(theta)
        
        
        
        if time.time() - start > till :
            cv2.destroyAllWindows()
            break
        
        if o == 1:
            cv2.destroyAllWindows()
            break

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
