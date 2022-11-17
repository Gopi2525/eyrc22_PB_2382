'''
*****************************************************************************************
*
*        =================================================
*             Pharma Bot Theme (eYRC 2022-23)
*        =================================================
*                                                         
*  This script is intended for implementation of Task 2A   
*  of Pharma Bot (PB) Theme (eYRC 2022-23).
*
*  Filename:			task_2a.py
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
# Filename:			task_2a.py
# Functions:		control_logic, detect_distance_sensor_1, detect_distance_sensor_2
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
##############################################################

def control_logic(sim):
	"""
	Purpose:
	---
	This function should implement the control logic for the given problem statement
	You are required to actuate the rotary joints of the robot in this function, such that
	it traverses the points in given order

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
	"""
	/Diff_Drive_Bot/Diff_Drive_Bot_Visible/left_joint/left_wheel
	/Diff_Drive_Bot/Diff_Drive_Bot_Visible/right_joint/right_wheel
	print(jointHandle_l)
	print(jointHandle_r)
	"""


	while 1 :

		#proximity sensor max range 3.5

		ds1 = detect_distance_sensor_1(sim)
		ds2 = detect_distance_sensor_2(sim)
		print("ds1",ds1)
		print("ds2",ds2)

		if ds1 == 0.0 :
			ds1 = 666
		else :
			ds1 == ds1

		if ds1 <= 0.2  :

			print("turn")

			#vel = 0
			#sim.setJointTargetVelocity(jointHandle_l, vel)
			#sim.setJointTargetVelocity(jointHandle_r, vel)

			if ds2 != 0:

				#rotate left
				vel = 0.5
				sim.setJointTargetVelocity(jointHandle_l, -vel)
				sim.setJointTargetVelocity(jointHandle_r, vel)
				time.sleep(2.8)

				vel = 0
				sim.setJointTargetVelocity(jointHandle_l, vel)
				sim.setJointTargetVelocity(jointHandle_r, vel)

			else :

				#rotate right
				vel = 0.5
				sim.setJointTargetVelocity(jointHandle_l, vel)
				sim.setJointTargetVelocity(jointHandle_r, -vel)
				time.sleep(2.8)

				vel = 0
				sim.setJointTargetVelocity(jointHandle_l, vel)
				sim.setJointTargetVelocity(jointHandle_r, vel)


		else :

			print("straight")

			a = 0.1650
			b = 0.2000

			if (ds2 >= a and ds2 <= b):

				vel = 0.5
				sim.setJointTargetVelocity(jointHandle_l, vel)
				sim.setJointTargetVelocity(jointHandle_r, vel)
				
			elif ds2 <= a :

				vel = 0.25
				sim.setJointTargetVelocity(jointHandle_l, -vel)
				sim.setJointTargetVelocity(jointHandle_r, vel)

				vel = 0.5
				sim.setJointTargetVelocity(jointHandle_l, vel)
				sim.setJointTargetVelocity(jointHandle_r, vel)
				

			elif ds2 >= b :

				vel = 0.25
				sim.setJointTargetVelocity(jointHandle_l, vel)
				sim.setJointTargetVelocity(jointHandle_r, -vel)
				
				vel = 0.5
				sim.setJointTargetVelocity(jointHandle_l, vel)
				sim.setJointTargetVelocity(jointHandle_r, vel)



	##################################################

def detect_distance_sensor_1(sim):
	"""
	Purpose:
	---
	Returns the distance of obstacle detected by proximity sensor named 'distance_sensor_1'

	Input Arguments:
	---
	`sim`    :   [ object ]
		ZeroMQ RemoteAPI object

	Returns:
	---
	distance  :  [ float ]
	    distance of obstacle from sensor

	Example call:
	---
	distance_1 = detect_distance_sensor_1(sim)
	"""
	distance = None
	##############  ADD YOUR CODE HERE  ##############

	sensorHandle_s1=sim.getObject("/distance_sensor_1")
	result , distance, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector=sim.readProximitySensor(sensorHandle_s1)


	##################################################
	return distance

def detect_distance_sensor_2(sim):
	"""
	Purpose:
	---
	Returns the distance of obstacle detected by proximity sensor named 'distance_sensor_2'

	Input Arguments:
	---
	`sim`    :   [ object ]
		ZeroMQ RemoteAPI object

	Returns:
	---
	distance  :  [ float ]
	    distance of obstacle from sensor

	Example call:
	---
	distance_2 = detect_distance_sensor_2(sim)
	"""
	distance = None
	##############  ADD YOUR CODE HERE  ##############

	sensorHandle_s2=sim.getObject("/distance_sensor_2")
	result , distance, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector=sim.readProximitySensor(sensorHandle_s2)

	##################################################
	return distance

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