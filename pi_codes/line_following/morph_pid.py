import numpy as np
import cv2
import time
import RPi.GPIO as GPIO
import sys
import datetime
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
from threading import Thread,Event
from picamera.array import PiRGBArray
from picamera import PiCamera
import math

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

        self.speed = 0
        self.base_pwm_pulse = 0
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

    def base_pwm(self,s):

        self.base_pwm_pulse = s

    def pwm_output(self,sensor,k):

        sensor = math.radians(sensor)

        self.error=self.sp-sensor
        self.integral = self.integral + (self.error*0.1)
        self.derivative = (self.error - self.error_past) / 0.1
        self.signal = (self.p*self.error + self.i*self.integral + self.d*self.derivative)
        self.error_past=self.error
        
        if k == 1 :
            self.speed = self.base_pwm_pulse - self.signal
            
        if k == 0 :
            self.speed = self.base_pwm_pulse + self.signal
        
        if self.speed > self.pwm_max :
            self.speed = self.pwm_max

        if self.speed < self.pwm_min :
            self.speed = self.pwm_min

        return self.speed
    
def set_wheel_pwm(left_pwm,right_pwm):
    
    GPIO.output(in1,GPIO.HIGH)
    GPIO.output(in2,GPIO.LOW)
    GPIO.output(in3,GPIO.LOW)
    GPIO.output(in4,GPIO.HIGH)
    
    print("left : ",left_pwm,"right :",right_pwm)
    
    pi_pwm_left.ChangeDutyCycle(left_pwm)
    pi_pwm_right.ChangeDutyCycle(right_pwm)

# Homomorphic filter class
class HomomorphicFilter:

    def __init__(self, a = 0.5, b = 1.5):
        self.a = float(a)
        self.b = float(b)
    # Filters
    def __butterworth_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = 1/(1+(Duv/filter_params[0]**2)**filter_params[1])
        return (1 - H)
    def __gaussian_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        H = np.zeros(I_shape)
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = np.exp((-Duv/(2*(filter_params[0])**2)))
        return (1 - H)
    # Methods
    def __apply_filter(self, I, H):
        H = np.fft.fftshift(H)
        I_filtered = (self.a + self.b*H)*I
        return I_filtered
    def filter(self, I, filter_params, filter='butterworth', H = None):
        if len(I.shape) != 2:
            raise Exception('Improper image')
        I_log = np.log1p(np.array(I, dtype="float"))
        I_fft = np.fft.fft2(I_log)
        if filter=='butterworth':
            H = self.__butterworth_filter(I_shape = I_fft.shape, filter_params = filter_params)
        elif filter=='gaussian':
            H = self.__gaussian_filter(I_shape = I_fft.shape, filter_params = filter_params)
        elif filter=='external':
            print('external')
            if len(H.shape) != 2:
                raise Exception('Invalid external filter')
        else:
            raise Exception('Selected filter not implemented')
        I_fft_filt = self.__apply_filter(I = I_fft, H = H)
        I_filt = np.fft.ifft2(I_fft_filt)
        I = np.exp(np.real(I_filt))-1
        return np.uint8(I)
# End of class HomomorphicFilter


def preprocessed(img):
    
    image = img.copy()
    
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    homo_filter = HomomorphicFilter(a = 5, b = 50)
    img_filtered = homo_filter.filter(I=img, filter_params=[60,2])

    return img_filtered

def centroid_detection(image,img_cen):
    
    ### centroid detection ###
    img = image.copy()
    img2 = img_cen.copy()
    
    ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
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
    
    return img2,theta

def blackdetector(img):

    lower_bound = np.array([0, 0, 0])
    upper_bound = np.array([20,20,20])
    black = cv2.inRange(img, lower_bound, upper_bound)
    n_black_pix = np.sum(black == 255)

    percentage_of_black_pixels = (n_black_pix/306720)*100
    if percentage_of_black_pixels != 0 :
        c = 1
    else:
        c = 0
    return c

def input_image_path_generator(image):
    
    img_cen = image.copy()
    img_blk = image.copy()
    
    img_filtered = preprocessed(image)
    navigation_image,theta = centroid_detection(img_filtered,img_cen)
    
    return navigation_image,theta

def image_processing_unit():

    camera = PiCamera()
    camera.brightness = 50
    camera.framerate = 15
    camera.exposure_mode = 'antishake'
    camera.resolution = (640, 480)
    rawCapture = PiRGBArray(camera, size=(640, 480))
    stream = camera.capture_continuous(rawCapture,format="bgr", use_video_port=True)
    
    for frame in stream:
        
        image = frame.array
        img_1 = image.copy()
    
        navi_dat,theta = input_image_path_generator(img_1)
        
        print(theta)
        
        left_pwm = bot_l.pwm_output(theta,0)
        right_pwm = bot_r.pwm_output(theta,1)
        set_wheel_pwm(left_pwm,right_pwm)

        cv2.imshow("navigation_image", navi_dat)
        
        key = cv2.waitKey(1) & 0xFF
        rawCapture.truncate(0)
        
        if key == ord("q"):
            
            pi_pwm_right.ChangeDutyCycle(0)
            pi_pwm_left.ChangeDutyCycle(0)
            
            cv2.destroyAllWindows()
            break
        
        if event.is_set():
            
            print("Stopped")
            
            pi_pwm_right.ChangeDutyCycle(0)
            pi_pwm_left.ChangeDutyCycle(0)
            
            cv2.destroyAllWindows()
            break

        
def control_logic():
    
    global bot_l,bot_r,event

    bot_l = pid()
    bot_r = pid()
    bot_l.setPID(2.5,0.01,0)
    bot_r.setPID(3.0,0.01,0)
    bot_l.setSetPoint(0)
    bot_r.setSetPoint(0)
    
    bot_l.base_pwm(35)
    bot_r.base_pwm(30)
    
    bot_l.set_min_max_pwm(0,50)
    bot_r.set_min_max_pwm(0,50)
    
    event = Event()
    Thread1 = Thread(target = image_processing_unit)
    Thread1.start()

    time.sleep(10)
    print('Main stopping thread')

    event.set()
    time.sleep(0.2)
    
    Thread1.join()
        
control_logic()

