import time
import RPi.GPIO as GPIO
import datetime 


def setup():
    
    global in1,in2,ena,in3,in4,enb,encoder_right
    
    in1=12
    in2=13
    ena=6
    in3=20
    in4=21
    enb=26
    
    encoder_right = 8

    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    GPIO.setup(in1,GPIO.OUT)
    GPIO.setup(in2,GPIO.OUT)
    GPIO.setup(ena,GPIO.OUT)   
    GPIO.setup(in3,GPIO.OUT)
    GPIO.setup(in4,GPIO.OUT)
    GPIO.setup(enb,GPIO.OUT)
    
    GPIO.setup(encoder_right, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    
setup()

GPIO.output(in3,GPIO.LOW)
GPIO.output(in4,GPIO.HIGH)

pi_pwm = GPIO.PWM(enb,1000)
pi_pwm.start(0)

t1 = datetime.datetime.now()

Kp = 1

pressed = 0
vel_set = 0.08
vel = 0

while 1:
    
    error = vel_set-vel
    output  =  Kp*error
    
    input_pwm = (output-0.0048)/(0.000052)
    
    duty = input_pwm
    #print("vel_set:",vel_set,"vel :",vel,"error:",error,"-duty:",duty,end = "\r")
    
    print("vel_set:",vel_set,"vel :",vel,"error:",error,"-duty:",duty)
    
    if error<0 :
        continue
    pi_pwm.ChangeDutyCycle(duty)
    if not GPIO.input(encoder_right):
            if not pressed:
                t2 = datetime.datetime.now()
                Vcm = 0.064*(0.226893/((t2-t1).total_seconds()))
                vel = Vcm
                t1 = t2
                pressed = 1
    else:
        pressed = 0
    time.sleep(0.001)
    

