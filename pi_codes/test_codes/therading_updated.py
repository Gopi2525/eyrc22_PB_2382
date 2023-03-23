from threading import Thread
import RPi.GPIO as GPIO
import time
import datetime

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
    
def move_bot(t1):
    
    print("Inside movebot function")
    
    in1=12
    in2=13
    ena=6
    in3=20
    in4=21
    enb=26

    GPIO.setup(in1,GPIO.OUT)
    GPIO.setup(in2,GPIO.OUT)
    GPIO.setup(in3,GPIO.OUT)
    GPIO.setup(in4,GPIO.OUT)
    GPIO.setup(ena,GPIO.OUT)
    GPIO.setup(enb,GPIO.OUT)
    
    GPIO.output(in1,0)
    GPIO.output(in2,0)
    GPIO.output(in3,0)
    GPIO.output(in4,0)
    GPIO.output(ena,0)
    GPIO.output(enb,0)
    
    GPIO.output(in1,GPIO.HIGH)
    GPIO.output(in2,GPIO.LOW)
    GPIO.output(in3,GPIO.LOW)
    GPIO.output(in4,GPIO.HIGH)
    
    pi_pwm_right = GPIO.PWM(enb,1000)
    pi_pwm_right.start(0)
    
    pi_pwm_left = GPIO.PWM(ena,1000)
    pi_pwm_left.start(0)
    
    duty_left = 73
    duty_right = 70
    
    pi_pwm_right.ChangeDutyCycle(duty_right)
    pi_pwm_left.ChangeDutyCycle(duty_left)
        
    while(1):

        if (datetime.datetime.now()- t1).total_seconds() >= 5:
            
            print(encoder_left_function.Vcm_left,encoder_right_function.Vcm_right)
            
            pi_pwm_right.ChangeDutyCycle(0)
            pi_pwm_left.ChangeDutyCycle(0)
                        
            return
            

def encoder_right_function(t1):
    
    print("inside right encoder function")
    encoder_right = 8
    encoder_right_function.Vcm_right = 0
    
    GPIO.setup(encoder_right, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    pressed = 0
    i = 0
    while 1:
            
        if not GPIO.input(encoder_right):
                if not pressed:
                    t2 = datetime.datetime.now()
                    encoder_right_function.Vcm_right = 0.064*(0.226893/((t2-t1).total_seconds()))
                    t1 = t2
                    pressed = 1
        else:
            pressed = 0
            
        time.sleep(0.1)
        
def encoder_left_function(t1):
    
    print("inside left encoder function")
    encoder_left = 7
    encoder_left_function.Vcm_left = 0
    
    GPIO.setup(encoder_left, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    t1 = datetime.datetime.now()

    pressed = 0
    while 1:
            
        if not GPIO.input(encoder_left):
                if not pressed:
                    t2 = datetime.datetime.now()
                    encoder_left_function.Vcm_left = 0.064*(0.226893/((t2-t1).total_seconds()))
                    t1 = t2
                    pressed = 1
        else:
            pressed = 0
            
        time.sleep(0.1)


GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
    
t1 = datetime.datetime.now()

Thread1 = Thread(target = encoder_right_function,args=(t1,),daemon = True)
Thread2 = Thread(target = encoder_left_function,args=(t1,),daemon = True)

Thread1.start()
Thread2.start()

time.sleep(2)

move_bot(t1)
    
print("code executed")

