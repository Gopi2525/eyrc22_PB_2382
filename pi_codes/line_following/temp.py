    elif command == 2:
        
        print("Align bot left new ")
        
        b = 13.1         #wheel_base
        r = 6.3/2        #wheel_radius
        Theta = 0.226893 #13degree
        s = r*Theta
        pulses_count  = int((b*3.14*theta)/(2*s*180))-1
        
        GPIO.output(in1,GPIO.LOW)
        GPIO.output(in2,GPIO.HIGH)
        GPIO.output(in3,GPIO.LOW)
        GPIO.output(in4,GPIO.HIGH)
        
        duty_left = 70
        duty_right = 70
        
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
            
    elif command == -2:
        
        print("Align bot right new ")
        
        b = 13.1         #wheel_base
        r = 6.3/2        #wheel_radius
        Theta = 0.226893 #13degree
        s = r*Theta
        pulses_count  = int((-1*b*3.14*theta)/(2*s*180))-1
        
        GPIO.output(in1,GPIO.HIGH)
        GPIO.output(in2,GPIO.LOW)
        GPIO.output(in3,GPIO.HIGH)
        GPIO.output(in4,GPIO.LOW)
        
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

