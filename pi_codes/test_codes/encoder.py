import time
import RPi.GPIO as GPIO
import datetime 

t1 = datetime.datetime.now()

BUTTON_GPIO = 8
if __name__ == '__main__':
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    pressed = 0
    i = 0
    while 1:        
        if not GPIO.input(BUTTON_GPIO):
            if not pressed:
                print(i)
                i=i+1
                t2 = datetime.datetime.now()
                Vcm = 0.064*(0.226893/((t2-t1).total_seconds()))
                print(Vcm)
                t1 = t2
                pressed = 1
        else:
            pressed = 0
        time.sleep(0.0001)
                
