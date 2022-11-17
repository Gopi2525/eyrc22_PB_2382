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

# Team ID:			[ PB_2382 ]
# Author List:		[ Aakash K , Devaprasad S , Gopi M , Ilam Thendral R ]
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

class Storage():

    def command_past():

        Storage.cmds = [0]

    def blocks_past():

        Storage.blks = [0]

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

			if (theta >= 1.22173 and theta <= 1.91986) or (theta >= 4.36332 and theta <= 5.06145 ) : # [70 to 110] and [250 to 290]

				command = 1 #straight
				block = "node"
				image = cv2.putText(img, 'Approaching node : straight', (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (55,55, 0), 1, cv2.LINE_AA)

			elif (theta >= 0 and theta <= 0.0523599) or (theta >= 3.08923 and theta <= 3.14159 ) or (theta >= 3.14159 and theta <= 3.19395 ) or (theta >= 6.23083 and theta <= 6.28319 ) or (): # [0 to 3] or [177 to 180] or [180 to 183] to [357 to 360]

				cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
				image = cv2.putText(img, 'Correct path : straight', (250, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0, 255, 0), 1, cv2.LINE_AA)
				print("straight")
				block = "Correct path"
				command = 1 #straight

			elif (theta >= 0.0523599 and theta <= 1.22173) or (theta >= 3.19395 and theta <= 4.36332) : # [3 to 70] and [183 to 250]

				cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
				image = cv2.putText(img, 'Come to lane : align right ', (250, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0, 0, 255), 1, cv2.LINE_AA)
				print("align right ")
				block = "align right"
				command = 3 #align right

			elif (theta >= 5.06145 and theta <= 6.23083) or (theta >= 1.91986 and theta <= 3.08923) : # [290 to 357] and [110 to 177]

				cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
				image = cv2.putText(img, 'Come to lane : align left ', (250, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0, 0, 255), 1, cv2.LINE_AA)
				print("align left ")
				block = "align left"
				command = 2 #align left 

			else :

				print("else")
				block = "else"
				command = 2

			"""


			if theta >= 1.48353 and theta <= 1.65806 : # 85 to 95

				cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

			if (theta >= 0.96 and theta <= 1.48353) or (theta >= 1.65806 and theta <= 2.18199 ) : # [55 to 85] and [95 to 125]

				cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

			else :

				cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

			"""


		return img,lines,command,block

	else :

		lines = np.ndarray([0,0])
		command = Storage.cmds[-1]
		print("None type")
		block = "None type"

		return img,lines,command,block

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

	Storage.command_past()
	Storage.blocks_past()

	#fourcc = cv2.VideoWriter_fourcc(*'XVID')
	#out = cv2.VideoWriter('output.mp4', fourcc, 20.0,(512,512))

	print("Align back to normal")
	vel = 0.15

	sim.setJointTargetVelocity(jointHandle_l, vel)
	sim.setJointTargetVelocity(jointHandle_r, -vel)

	time.sleep(2)

	while 1 :

		img, resX, resY = sim.getVisionSensorCharImage(visionSensorHandle)
		img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
		img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)

		frame , l , c , b = givelines(img)
		cv2.imshow('Frame', frame)
		#out.write(frame)

		Storage.cmds.append(c)
		Storage.blks.append(b)

		if len(Storage.blks) >= 400 and Storage.blks[-175::1].count("node") >= 100 :

			vel = 0

			sim.setJointTargetVelocity(jointHandle_l, vel)
			sim.setJointTargetVelocity(jointHandle_r, vel)

			Storage.blks = [0]

			time.sleep(5)

			print("Reached node")

		if c == 0 :

			#stop

			vel = 0

			sim.setJointTargetVelocity(jointHandle_l, vel)
			sim.setJointTargetVelocity(jointHandle_r, vel)

		elif c == 1 :

			#straight

			vel = 0.05

			sim.setJointTargetVelocity(jointHandle_l, vel)
			sim.setJointTargetVelocity(jointHandle_r, vel)

		elif c == 2 :

			#left

			vel = 0.05

			sim.setJointTargetVelocity(jointHandle_l, -vel)
			sim.setJointTargetVelocity(jointHandle_r, vel)

		elif c == 3 :

			#right

			vel = 0.05

			sim.setJointTargetVelocity(jointHandle_l, vel)
			sim.setJointTargetVelocity(jointHandle_r, -vel)

		elif (c == 20) or (c == 30) :

			if c == 20:

				#turn left

				vel = 0.05

				sim.setJointTargetVelocity(jointHandle_l, -vel)
				sim.setJointTargetVelocity(jointHandle_r, vel)

			elif c == 30:

				#turn right

				vel = 0.05

				sim.setJointTargetVelocity(jointHandle_l, vel)
				sim.setJointTargetVelocity(jointHandle_r, -vel)


		if cv2.waitKey(1) & 0xFF == ord('q'):

			print(Storage.blks)

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