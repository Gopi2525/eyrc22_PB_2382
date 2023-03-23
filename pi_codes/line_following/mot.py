import RPi.GPIO as GPIO
import time
from AlphaBot import AlphaBot

Ab = AlphaBot()
"""
left = -50
right = 65
Ab.setMotor(left, right)
time.sleep(5)
"""

Ab.stop()