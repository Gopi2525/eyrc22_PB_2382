'''
*****************************************************************************************
*
*        =================================================
*             Pharma Bot Theme (eYRC 2022-23)
*        =================================================
*                                                         
*  This script is intended for implementation of Task 2B   
*  of Pharma Bot (PB) Theme (eYRC 2022-23).
*
*  Filename:			task_2b.py
*  Created:				
*  Last Modified:		8/10/2022
*  Author:				e-Yantra Team
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:			[ Team-ID ]
# Author List:		[ Names of team members worked on this file separated by Comma: Name1, Name2, ... ]
# Filename:			task_2b.py
# Functions:		control_logic, read_qr_code
# 					[ Comma separated list of functions in this file ]
# Global variables:	
# 					[ List of global variables defined in this file ]

####################### IMPORT MODULES #######################
## You are not allowed to make any changes in this section. ##
##############################################################
import  sys
import traceback
import time
import os
import math
from zmqRemoteApi import RemoteAPIClient
import zmq
import numpy as np
import cv2
import random
from pyzbar.pyzbar import decode
##############################################################

################# ADD UTILITY FUNCTIONS HERE #################

def givelines(img):

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray, 50, 150, apertureSize=3)
	lines = cv2.HoughLines(edges, 1, np.pi/180, 200)


	if isinstance(lines, np.ndarray) :

		for r_theta in lines:

			arr = np.array(r_theta[0], dtype=np.float64)
			r, theta = arr

			a = np.cos(theta)
			b = np.sin(theta)

			x0 = a*r
			y0 = b*r

			x1 = int(x0 + 1000*(-b))
			y1 = int(y0 + 1000*(a))

			x2 = int(x0 - 1000*(-b))
			y2 = int(y0 - 1000*(a))

			cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
			image = cv2.putText(img, 'All lines', (250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (255, 0, 0), 1, cv2.LINE_AA)

			if (theta >= 1.22173 and theta <= 1.5708) or (theta >= 4.71239 and theta <= 5.23599 ) : # [90 to 70] and [270 to 300]

				command = 1 #straight
				image = cv2.putText(img, 'Approaching node', (250, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (55,55, 0), 1, cv2.LINE_AA)

			elif (theta >= 0 and theta <= 0.0523599) or (theta >= 6.23083 and theta <= 6.28319 ) : # [3 to 0] and [0 to 357]

				cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
				image = cv2.putText(img, 'Correct path', (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0, 255, 0), 1, cv2.LINE_AA)
				print("straight")
				command = 1 #straight

			elif (theta >= 0.0523599 and theta <= 0.785398) or (theta >= 5.49779 and theta <= 6.28319) : # [45 to 3] and [270 to 357]

				cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
				image = cv2.putText(img, 'Come to lane', (250, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0, 0, 255), 1, cv2.LINE_AA)
				print("turn")
				command = 0 #rotate

			else :

				print("else")
				command = 0

			"""


			if theta >= 1.48353 and theta <= 1.65806 : # 85 to 95

				cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

			if (theta >= 0.96 and theta <= 1.48353) or (theta >= 1.65806 and theta <= 2.18199 ) : # [55 to 85] and [95 to 125]

				cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

			else :

				cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

			"""


		return img,lines,command

	else :

		lines = np.ndarray([0,0])
		command = 0
		print("None type")

		return img,lines,command

##############################################################

def control_logic(sim):
	"""
	Purpose:
	---
	This function should implement the control logic for the given problem statement
	You are required to make the robot follow the line to cover all the checkpoints
	and deliver packages at the correct locations.

	Input Arguments:
	---
	`sim`    :   [ object ]
		ZeroMQ RemoteAPI object

	Returns:
	---
	None

	Example call:
	---
	control_logic(sim)
	"""
	##############  ADD YOUR CODE HERE  ##############

	jointHandle_l=sim.getObject("/left_joint")
	jointHandle_r=sim.getObject("/right_joint")
	visionSensorHandle = sim.getObject('/vision_sensor')
	defaultIdleFps = sim.getInt32Param(sim.intparam_idle_fps)
	sim.setInt32Param(sim.intparam_idle_fps, 0)


	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter('output.mp4', fourcc, 20.0,(512,512))

	while 1 :

		img, resX, resY = sim.getVisionSensorCharImage(visionSensorHandle)
		img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
		img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)

		frame , l ,c = givelines(img)
		cv2.imshow('Frame', frame)
		out.write(frame)

		if c == 1 :

			vel = 0.1

			sim.setJointTargetVelocity(jointHandle_l, vel)
			sim.setJointTargetVelocity(jointHandle_r, vel)

		elif c == 0 :

			vel = 0.05

			sim.setJointTargetVelocity(jointHandle_l, vel)
			sim.setJointTargetVelocity(jointHandle_r, -vel)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break



	##################################################

def read_qr_code(sim):
	"""
	Purpose:
	---
	This function detects the QR code present in the camera's field of view and
	returns the message encoded into it.

	Input Arguments:
	---
	`sim`    :   [ object ]
		ZeroMQ RemoteAPI object

	Returns:
	---
	`qr_message`   :    [ string ]
		QR message retrieved from reading QR code

	Example call:
	---
	control_logic(sim)
	"""
	qr_message = None
	##############  ADD YOUR CODE HERE  ##############

	##################################################
	return qr_message


######### YOU ARE NOT ALLOWED TO MAKE CHANGES TO THE MAIN CODE BELOW #########

if __name__ == "__main__":
	client = RemoteAPIClient()
	sim = client.getObject('sim')	

	try:

		## Start the simulation using ZeroMQ RemoteAPI
		try:
			return_code = sim.startSimulation()
			if sim.getSimulationState() != sim.simulation_stopped:
				print('\nSimulation started correctly in CoppeliaSim.')
			else:
				print('\nSimulation could not be started correctly in CoppeliaSim.')
				sys.exit()

		except Exception:
			print('\n[ERROR] Simulation could not be started !!')
			traceback.print_exc(file=sys.stdout)
			sys.exit()

		## Runs the robot navigation logic written by participants
		try:
			time.sleep(5)
			control_logic(sim)

		except Exception:
			print('\n[ERROR] Your control_logic function throwed an Exception, kindly debug your code!')
			print('Stop the CoppeliaSim simulation manually if required.\n')
			traceback.print_exc(file=sys.stdout)
			print()
			sys.exit()

		
		## Stop the simulation using ZeroMQ RemoteAPI
		try:
			return_code = sim.stopSimulation()
			time.sleep(0.5)
			if sim.getSimulationState() == sim.simulation_stopped:
				print('\nSimulation stopped correctly in CoppeliaSim.')
			else:
				print('\nSimulation could not be stopped correctly in CoppeliaSim.')
				sys.exit()

		except Exception:
			print('\n[ERROR] Simulation could not be stopped !!')
			traceback.print_exc(file=sys.stdout)
			sys.exit()

	except KeyboardInterrupt:
		## Stop the simulation using ZeroMQ RemoteAPI
		return_code = sim.stopSimulation()
		time.sleep(0.5)
		if sim.getSimulationState() == sim.simulation_stopped:
			print('\nSimulation interrupted by user in CoppeliaSim.')
		else:
			print('\nSimulation could not be interrupted. Stop the simulation manually .')
			sys.exit()