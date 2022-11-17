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

def tracker(img):

	a = 25

	cv2.rectangle(img, pt1=(256-a,0), pt2=(256+a,512), color=(0,255,0), thickness=2)
	return img

def track_result(centroid):

	x,y = centroid
	a = 25


	if x > 255 - a and x < 255 + a :

		print("in position")
		command = 1

		return 0,command

	else :

		print("come")

		if x < 256-a :

			print("align right")
			command = 3

		else :

			print("align left")
			command = 2
	
		return 1,command

def execute_command(sim, command):

	jointHandle_l=sim.getObject("/left_joint")
	jointHandle_r=sim.getObject("/right_joint")

	if command == 0 :

		#stop
		vel = 0
		sim.setJointTargetVelocity(jointHandle_l, vel)
		sim.setJointTargetVelocity(jointHandle_r, vel)

		return None

	elif command == 1 :

		#straight
		vel = 1.5
		sim.setJointTargetVelocity(jointHandle_l, vel)
		sim.setJointTargetVelocity(jointHandle_r, vel)

		return None

	elif command == 2 :

		#left
		vel = 0.15
		sim.setJointTargetVelocity(jointHandle_l, -vel)
		sim.setJointTargetVelocity(jointHandle_r, vel)

		return None

	elif command == 3 :

		#right
		vel = 0.15
		sim.setJointTargetVelocity(jointHandle_l, vel)
		sim.setJointTargetVelocity(jointHandle_r, -vel)

		return None

def execute_state_command(sim,img2,t,jointHandle_l,jointHandle_r):

	if t == 1 :

		print("turn")
		print(Storage.cmds_nodes[Storage.cmds_nodes_state])
		change_new_lane(sim,Storage.cmds_nodes[Storage.cmds_nodes_state],img2)
		Storage.cmds_nodes_state = Storage.cmds_nodes_state + 1

		return 1

	else :

		print("Continue")

		return 1

def givetheta(img):

	grayimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)	
	(thresh, blackAndWhiteImage) = cv2.threshold(grayimage, 127, 255, cv2.THRESH_BINARY)
	edges = cv2.Canny(blackAndWhiteImage, 50, 150, apertureSize=3)

	kernel = np.array([[0, -1, 0],[-1, 5,-1],[0, -1, 0]])
	image_sharp = cv2.filter2D(src=edges, ddepth=-1, kernel=kernel)

	lines = cv2.HoughLines(image_sharp, 1, np.pi/180, 200)

	rl = math.radians(1)
	ru = math.radians(75)

	ll = math.radians(105)
	lu = math.radians(179)

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

			if (theta >= rl and theta <= ru) or (theta >= math.degrees(180)+rl and theta <= math.degrees(180)+ru) :

				command = 3 #align right	

				return command

			elif (theta >= ll and theta <= lu) or (theta >= math.degrees(180)+ll and theta <= math.degrees(180)+lu):

				command = 2 #align left 

				return command

			elif (theta >= 0 and theta <= rl) or (theta >= lu and theta <= 3.14159 ) or (theta >= 3.14159 and theta <= math.degrees(180)+rl ) or (theta >= math.degrees(180)+lu and theta <= 6.28319 ): # [0 to 3] or [177 to 180] or [180 to 183] to [357 to 360]

				command = 1 #straight

				return command

			else :

				command = Storage.cmds[-1]

				return command

	else :

		command = Storage.cmds[-1]

		return command

def givelines(img):

	grayimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)	
	(thresh, blackAndWhiteImage) = cv2.threshold(grayimage, 127, 255, cv2.THRESH_BINARY)
	edges = cv2.Canny(blackAndWhiteImage, 50, 150, apertureSize=3)

	kernel = np.array([[0, -1, 0],[-1, 5,-1],[0, -1, 0]])
	image_sharp = cv2.filter2D(src=edges, ddepth=-1, kernel=kernel)

	lines = cv2.HoughLines(image_sharp, 1, np.pi/180, 200)

	rl = math.radians(1)
	ru = math.radians(75)

	ll = math.radians(105)
	lu = math.radians(179)

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

			#cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
			#image = cv2.putText(img, 'All lines', (250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (255, 0, 0), 1, cv2.LINE_AA)

			if (theta >= rl and theta <= ru) or (theta >= math.degrees(180)+rl and theta <= math.degrees(180)+ru) :

				#print("align right")
				cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
				image = cv2.putText(img, 'Come to lane : align right ', (250, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0, 0, 255), 1, cv2.LINE_AA)
				block = "align right"
				command = 3 #align right	

				return img,lines,command,block

			elif (theta >= ll and theta <= lu) or (theta >= math.degrees(180)+ll and theta <= math.degrees(180)+lu):

				#print("align lef")
				cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
				image = cv2.putText(img, 'Come to lane : align left ', (250, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0, 0, 255), 1, cv2.LINE_AA)
				block = "align left"
				command = 2 #align left 

				return img,lines,command,block

			elif (theta >= 0 and theta <= rl) or (theta >= lu and theta <= 3.14159 ) or (theta >= 3.14159 and theta <= math.degrees(180)+rl ) or (theta >= math.degrees(180)+lu and theta <= 6.28319 ): # [0 to 3] or [177 to 180] or [180 to 183] to [357 to 360]

				#print("straight")
				cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
				image = cv2.putText(img, 'Correct path : straight', (250, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0, 255, 0), 1, cv2.LINE_AA)
				block = "Correct path"
				command = 1 #straight

				return img,lines,command,block

			else :

				#print("else")
				block = "else"
				command = Storage.cmds[-1]



				return img,lines,command,block

	else :

		lines = np.ndarray([0,0])
		command = Storage.cmds[-1]
		print("None type")
		block = "None type"

		return img,lines,command,block

def centroid_finder(image):

	img = image

	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17) 

	lower_bound = np.array([74, 74, 74])
	upper_bound = np.array([160,160, 160])
	image = cv2.inRange(image, lower_bound, upper_bound)


	blur = cv2.GaussianBlur(image,(13,13),0)
	thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)[1]

	image = thresh

	blurImg = cv2.blur(image,(10,10)) 

	kernel = np.array([[0, -1, 0],[-1, 5,-1],[0, -1, 0]])
	image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
	image = image_sharp

	contours, hierarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	contours = sorted(contours, key = cv2.contourArea, reverse = True) [:1]

	for c in contours:
	   # calculate moments for each contour
	   M = cv2.moments(c)
	 
	   # calculate x,y coordinate of center
	   cX = int(M["m10"] / M["m00"])
	   cY = int(M["m01"] / M["m00"])
	   cv2.circle(img, (cX, cY), 5, (0, 255, 0), -1)
	   cv2.putText(img, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	   a = (cX, cY)

	return(img,a)

def change_new_lane(sim,cmds_nodes,img2):

	jointHandle_l=sim.getObject("/left_joint")
	jointHandle_r=sim.getObject("/right_joint")

	if cmds_nodes == 20:

		#turn left

		vel = 1
		sim.setJointTargetVelocity(jointHandle_l, -vel)
		sim.setJointTargetVelocity(jointHandle_r, vel)
		time.sleep(0.9)

		return None

	elif cmds_nodes == 30:

		#turn right
		vel = 1
		sim.setJointTargetVelocity(jointHandle_l, vel)
		sim.setJointTargetVelocity(jointHandle_r, -vel)
		time.sleep(1)

		return None

	elif cmds_nodes == 100 :


		return None

	elif cmds_nodes == 0 :

		Storage.endl = 1

		return None

def yellowdetector(img):

	lower_bound = np.array([0, 190, 230])
	upper_bound = np.array([30,220, 260])
	yellow = cv2.inRange(img, lower_bound, upper_bound)

	n_yellow_pix = np.sum(yellow == 255)

	return yellow, n_yellow_pix

def yellowchecker(sim,jointHandle_l,jointHandle_r):

	if Storage.yellow[-1] == 0 and Storage.yellow[-2] != 0 :

		vel = 0

		sim.setJointTargetVelocity(jointHandle_l, vel)
		sim.setJointTargetVelocity(jointHandle_r, vel)

		print("Stoped at node")

		return 1

	else :

		return 0

class Storage():

    def command_past():

        Storage.cmds = [0]

    def blocks_past():

        Storage.blks = [0]

    def n_yellow_pix():

    	Storage.yellow = [0,0,0,0,0,0,0,0]

    def n_black_pix():

    	Storage.black = [0]

    def commands():

    	Storage.cmds_nodes = [20 ,30 ,20 ,30, 100, 30, 20, 30, 100, 30, 20, 30, 100, 30, 20, 30, 0]
    	Storage.cmds_nodes_state = 0
    	Storage.cmds_checkpoints = ["checkpoint E","checkpoint I","checkpoint M"]
    	Storage.cmds_checkpoints_state = 0

    def end():
    	Storage.endl = 0

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

	##### geting handles
	jointHandle_l=sim.getObject("/left_joint")
	jointHandle_r=sim.getObject("/right_joint")
	visionSensorHandle = sim.getObject('/vision_sensor')
	defaultIdleFps = sim.getInt32Param(sim.intparam_idle_fps)
	sim.setInt32Param(sim.intparam_idle_fps, 0)

	##### initializing storage
	Storage.command_past()
	Storage.blocks_past()
	Storage.n_yellow_pix()
	Storage.commands()
	Storage.end()

	while 1 :

		##### geting input from vision sensor
		img, resX, resY = sim.getVisionSensorCharImage(visionSensorHandle)
		img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
		img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
		img2 = img

		##### sharpening image to avoid none type errors in hough transform 
		kernel = np.array([[0, -1, 0],[-1, 5,-1],[0, -1, 0]])
		image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
		img = image_sharp

		##### calling functions to execute task
		frame , lines , command , block = givelines(img)
		image , centroid = centroid_finder(frame)
		track_state , track_command = track_result(centroid)
		image = tracker(image)
		yellow, n_yellow_pix = yellowdetector(frame)
		state = yellowchecker(sim,jointHandle_l,jointHandle_r)
		operation = execute_state_command(sim,img2,state,jointHandle_l,jointHandle_r)

		execute_command(sim, track_command)
		execute_command(sim, command)

		if Storage.endl == 1 :

			vel = 0.0
			sim.setJointTargetVelocity(jointHandle_l, vel)
			sim.setJointTargetVelocity(jointHandle_r, vel)

			break

		##### storing previous datas
		Storage.cmds.append(command)
		Storage.blks.append(block)
		Storage.yellow.append(n_yellow_pix)

		cv2.imshow('Frame', frame)

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

	visionSensorHandle = sim.getObject('/vision_sensor')
	defaultIdleFps = sim.getInt32Param(sim.intparam_idle_fps)
	sim.setInt32Param(sim.intparam_idle_fps, 0)

	img, resX, resY = sim.getVisionSensorCharImage(visionSensorHandle)
	img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
	img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)

	data = decode(img)

	if len(data) != 0 :
		t = data[0].data
		qr_message = t.decode()

	else :

		qr_message = None

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