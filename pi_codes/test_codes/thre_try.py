from threading import Thread
import RPi.GPIO as GPIO
import time
import datetime 

def encoder_right(t1):
    
    print("inside right encoder function")
    encoder_right = 8
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    GPIO.setup(encoder_right, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    pressed = 0
    i = 0
    while 1:
        if not GPIO.input(encoder_right):
                if not pressed:
                    t2 = datetime.datetime.now()
                    Vcm = 0.064*(0.226893/((t2-t1).total_seconds()))
                    print("------------Vcm",Vcm)
                    t1 = t2
                    pressed = 1
        else:
            pressed = 0
            
        time.sleep(0.001)
        
def encoder_left(t1):
    
    print("inside left encoder function")
    encoder_left = 7
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    GPIO.setup(encoder_left, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    t1 = datetime.datetime.now()

    pressed = 0
    while 1:
        if not GPIO.input(encoder_left):
                if not pressed:
                    t2 = datetime.datetime.now()
                    Vcm = 0.064*(0.226893/((t2-t1).total_seconds()))
                    print("Vcm",Vcm)
                    t1 = t2
                    pressed = 1
        else:
            pressed = 0
            
        time.sleep(0.001)

t1 = datetime.datetime.now()

Thread1 = Thread(target = encoder_right,args=(t1,),daemon = True)
Thread2 = Thread(target = encoder_left,args=(t1,),daemon = True)

Thread1.start()
Thread2.start()

Thread1.join()
Thread1.join()