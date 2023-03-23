from threading import Thread
import RPi.GPIO as GPIO
import time
import datetime

def left_turn():
    
    b = 13.1         #wheel_base
    r = 6.3/2        #wheel_radius
    theta = 0.226893 #13degree
    s = r*theta
    pulses_count  = int((3.14*b)/(4*s))
    
    print("inside left turn function")
    encoder_right = 8
    in3=20
    in4=21
    enb=26
    
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    GPIO.setup(in3,GPIO.OUT)
    GPIO.setup(in4,GPIO.OUT)
    GPIO.setup(enb,GPIO.OUT)
    
    GPIO.output(in3,GPIO.LOW)
    GPIO.output(in4,GPIO.HIGH)
    
    pi_pwm = GPIO.PWM(enb,1000)
    pi_pwm.start(0)
    
    GPIO.setup(encoder_right, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    pressed = 0
    i = 0
    
    duty = 70
    pi_pwm.ChangeDutyCycle(duty)

    
    while 1:
        if not GPIO.input(encoder_right):
                if not pressed:
                    
                    if i == pulses_count+3:
                        
                        duty = 0
                        pi_pwm.ChangeDutyCycle(duty)
                        
                        break
                    
                    i = i+1
                    pressed = 1
        else:
            pressed = 0
            
        time.sleep(0.001)
        
    print("turned 90 degrees left")


def right_turn():
    
    b = 13.1         #wheel_base
    r = 6.3/2        #wheel_radius
    theta = 0.226893 #13degree
    s = r*theta
    pulses_count  = int((3.14*b)/(4*s))
    
    print("inside right turn function")
    encoder_left = 7
    in1=12
    in2=13
    ena=6
    
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    GPIO.setup(in1,GPIO.OUT)
    GPIO.setup(in2,GPIO.OUT)
    GPIO.setup(ena,GPIO.OUT)
    
    GPIO.output(in1,GPIO.HIGH)
    GPIO.output(in2,GPIO.LOW)
    
    pi_pwm = GPIO.PWM(ena,1000)
    pi_pwm.start(0)
    
    GPIO.setup(encoder_left, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    pressed = 0
    i = 0
    
    duty = 70
    pi_pwm.ChangeDutyCycle(duty)

    
    while 1:
        if not GPIO.input(encoder_left):
                if not pressed:
                    
                    if i == pulses_count+3:
                        
                        duty = 0
                        pi_pwm.ChangeDutyCycle(duty)
                        
                        break
                    
                    i = i+1
                    pressed = 1
        else:
            pressed = 0
            
        time.sleep(0.001)
        
    print("turned 90 degrees right")

left_turn()

#right_turn()
    