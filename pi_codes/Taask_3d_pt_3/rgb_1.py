import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
def led():
    redPin = 24
    gndPin = 23
    greenPin = 5
    bluePin = 18

    GPIO.setup(redPin,GPIO.OUT)
    GPIO.setup(gndPin,GPIO.OUT)
    GPIO.setup(greenPin,GPIO.OUT)
    GPIO.setup(bluePin,GPIO.OUT)
    
    global pi_pwm_r,pi_pwm_g,pi_pwm_b

    pi_pwm_r = GPIO.PWM(redPin,100)
    pi_pwm_g = GPIO.PWM(greenPin,100)
    pi_pwm_b = GPIO.PWM(bluePin,100)

    GPIO.output(gndPin,0)

    duty_r=50
    duty_g=50
    duty_b=100

    pi_pwm_r.start(duty_r)
    pi_pwm_g.start(duty_g)
    pi_pwm_b.start(duty_b)

led()
for i in range(1,1000000,1):
    pass

print("Ended")
GPIO.cleanup()