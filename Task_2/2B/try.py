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

def commands_at_node(sim,img,c):

	jointHandle_l=sim.getObject("/left_joint")
	jointHandle_r=sim.getObject("/right_joint")

	if c == 20:

		#turn left

		vel = 0.5

		sim.setJointTargetVelocity(jointHandle_l, -vel)
		sim.setJointTargetVelocity(jointHandle_r, vel)

		time.sleep(1.7)

		while 1 :

			sim.setJointTargetVelocity(jointHandle_l, -vel)
			sim.setJointTargetVelocity(jointHandle_r, vel)

			if givetheta(img) == 1 :

				break

		k = 0

	elif c == 30:

		#turn right

		vel = 0.5

		vel = 0.5

		sim.setJointTargetVelocity(jointHandle_l, vel)
		sim.setJointTargetVelocity(jointHandle_r, -vel)

		time.sleep(1.7)

		while 1 :

			sim.setJointTargetVelocity(jointHandle_l, vel)
			sim.setJointTargetVelocity(jointHandle_r, -vel)

			if givetheta(img) == 1 :

				break

		k = 0

	elif c == 1 :

		#straight

		vel = 0.5

		while 1 :

			sim.setJointTargetVelocity(jointHandle_l, vel)
			sim.setJointTargetVelocity(jointHandle_r, -vel)

			if givetheta(img) == 1 :

				break

		vel = 0.7

		sim.setJointTargetVelocity(jointHandle_l, vel)
		sim.setJointTargetVelocity(jointHandle_r, vel)

		k = 0

	elif c == 0 :

		k = 1

	return k

def givetheta(img):

	grayimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)	
	(thresh, blackAndWhiteImage) = cv2.threshold(grayimage, 127, 255, cv2.THRESH_BINARY)
	edges = cv2.Canny(blackAndWhiteImage, 50, 150, apertureSize=3)
	lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

	#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#edges = cv2.Canny(gray, 50, 150, apertureSize=3)
	#lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

	a = math.radians(0)
	b = math.radians(360)
	c = math.radians(180)
	d = math.radians(180)
	
	f = math.radians(88)
	g = math.radians(268)
	h = math.radians(290)
	i = math.radians(110)

	lower_bound = np.array([0, 190, 230])
	upper_bound = np.array([30,220, 260])
	yellow = cv2.inRange(img, lower_bound, upper_bound)

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

			if (theta >= a and theta <= f) or (theta >= d and theta <= g) : # [3 to 70] and [183 to 250]

				command = 3 #align right

			elif (theta >= h and theta <= b) or (theta >= i and theta <= c) : # [290 to 357] and [110 to 177]

				command = 2 #align left 

			elif (theta >= 0 and theta <= a) or (theta >= c and theta <= 3.14159 ) or (theta >= 3.14159 and theta <= d ) or (theta >= b and theta <= 6.28319 ) or (): # [0 to 3] or [177 to 180] or [180 to 183] to [357 to 360]

				command = 1 #straight

			else :

				command = 2

		return command

	else :

		command = Storage.cmds[-1]

		return command

class Storage():

    def command_past():

        Storage.cmds = [0]

    def blocks_past():

        Storage.blks = [0]

    def n_yellow_pix():

    	Storage.yellow = [0]

    def n_black_pix():

    	Storage.black = [0]

    def commands():

    	Storage.cmds_nodes = [20 ,30 ,20 ,30, 1, 30, 20, 30, 1, 30, 20, 30, 1, 30, 20, 30, 0]

def yellowdetector(img):

	lower_bound = np.array([0, 190, 230])
	upper_bound = np.array([30,220, 260])
	yellow = cv2.inRange(img, lower_bound, upper_bound)

	return yellow

def blackdetector(img):

	lower_bound = np.array([140, 140, 140])
	upper_bound = np.array([160,160, 160])
	black = cv2.inRange(img, lower_bound, upper_bound)

	return black

def givelines(img):

	grayimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)	
	(thresh, blackAndWhiteImage) = cv2.threshold(grayimage, 127, 255, cv2.THRESH_BINARY)
	edges = cv2.Canny(blackAndWhiteImage, 50, 150, apertureSize=3)
	lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

	a = math.radians(0)
	b = math.radians(360)
	c = math.radians(180)
	d = math.radians(180)
	
	f = math.radians(88)
	g = math.radians(268)
	h = math.radians(290)
	i = math.radians(110)

	lower_bound = np.array([0, 190, 230])
	upper_bound = np.array([30,220, 260])
	yellow = cv2.inRange(img, lower_bound, upper_bound)

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

			if (theta >= a and theta <= f) or (theta >= d and theta <= g) : # [3 to 70] and [183 to 250]

				cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
				image = cv2.putText(img, 'Come to lane : align right ', (250, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0, 0, 255), 1, cv2.LINE_AA)
				print("align right ")
				block = "align right"
				command = 3 #align right

			elif (theta >= h and theta <= b) or (theta >= i and theta <= c) : # [290 to 357] and [110 to 177]

				cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
				image = cv2.putText(img, 'Come to lane : align left ', (250, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0, 0, 255), 1, cv2.LINE_AA)
				print("align left ")
				block = "align left"
				command = 2 #align left 

			elif (theta >= 0 and theta <= a) or (theta >= c and theta <= 3.14159 ) or (theta >= 3.14159 and theta <= 3.19395 ) or (theta >= b and theta <= 6.28319 ) or (): # [0 to 3] or [177 to 180] or [180 to 183] to [357 to 360]

				cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
				image = cv2.putText(img, 'Correct path : straight', (250, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0, 255, 0), 1, cv2.LINE_AA)
				print("straight")
				block = "Correct path"
				command = 1 #straight

			else :

				print("else")
				block = "else"
				command = 2

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
	Storage.n_yellow_pix()
	Storage.n_black_pix()
	Storage.commands()

	t = 0
	k = 0

	#fourcc = cv2.VideoWriter_fourcc(*'XVID')
	#out = cv2.VideoWriter('output.mp4', fourcc, 20.0,(512,512))

	while 1 :

		img, resX, resY = sim.getVisionSensorCharImage(visionSensorHandle)
		img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
		img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
		#out.write(img)

		xp = [0, 64, 128, 192, 255]
		fp = [0, 16, 128, 240, 255]
		x = np.arange(256)
		table = np.interp(x, xp, fp).astype('uint8')
		cont_img = cv2.LUT(img, table)
		#cv2.imshow("Contrast streched", cont_img)

		kernel = np.array([[0, -1, 0],[-1, 5,-1],[0, -1, 0]])
		image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

		frame , l , c , b = givelines(image_sharp)
		y = yellowdetector(frame)
		bk = blackdetector(frame)

		cv2.imshow('Frame', frame)
		#cv2.imshow('Y', y)
		#cv2.imshow('Bk', bk)

		n_yellow_pix = np.sum(y == 255)
		n_black_pix = np.sum(bk == 255)

		Storage.yellow.append(n_yellow_pix)
		Storage.black.append(n_black_pix)
		Storage.cmds.append(c)
		Storage.blks.append(b)

		if Storage.yellow[-1] == 0 and Storage.yellow[-2] != 0 :

			vel = 0

			sim.setJointTargetVelocity(jointHandle_l, vel)
			sim.setJointTargetVelocity(jointHandle_r, vel)

			print("Stoped at node")

			c = Storage.cmds_nodes[t]
			t = t+1

			k = commands_at_node(sim,img,c)
			print(k)

		elif k == 1 :

			break

		elif c == 0 :

			#stop

			vel = 0

			sim.setJointTargetVelocity(jointHandle_l, vel)
			sim.setJointTargetVelocity(jointHandle_r, vel)

		elif c == 1 :

			#straight

			vel = 0.75

			sim.setJointTargetVelocity(jointHandle_l, vel)
			sim.setJointTargetVelocity(jointHandle_r, vel)

		elif c == 2 :

			#left

			vel = 0.02

			sim.setJointTargetVelocity(jointHandle_l, -vel)
			sim.setJointTargetVelocity(jointHandle_r, vel)


		elif c == 3 :

			#right

			vel = 0.02

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
			
			control_logic(sim)
			time.sleep(5)


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