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

# Team ID:			[ PB_2382 ]
# Author List:		[ Aakash K , Devaprasad S , Gopi M , Ilam Thendral R ]
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
	sensorHandle_s3=sim.getObject("/distance_sensor_3")

	S1 =45
	vel = 0
	sim.setJointTargetVelocity(jointHandle_l, vel)
	sim.setJointTargetVelocity(jointHandle_r, vel)
	way_points = 0
	k = 0


	while 1 :

		#proximity sensor max range 3.5

		ds1 = detect_distance_sensor_1(sim)
		ds2 = detect_distance_sensor_2(sim)
		result , ds3, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector=sim.readProximitySensor(sensorHandle_s3)

		print(f"ds1 : {ds1} ds2 : {ds2} ds3 : {ds3}")

		if ds2 != 0 and ds3 != 0 :

			s1 = math.sqrt((ds2*ds2)+(ds3*ds3)-(2*ds2*ds3*math.cos(S1)))
			S3 = math.degrees(math.asin((ds3/s1)*(math.sin(S1))))

			print(f"Orientation with right wall {S3}")

		elif ds3 == 0:

			print("Orientation canot be found")

		if ds1 == 0.0 :

			ds1 = 666

		else :

			ds1 == ds1

		if ds1 <= 0.3  :

			print("turn")

			vel = 0
			sim.setJointTargetVelocity(jointHandle_l, vel)
			sim.setJointTargetVelocity(jointHandle_r, vel)
			
			way_points = way_points + 1
			print(f"Way point = {way_points}")

			if way_points == 10 :

				print(f"Way point = {way_points}")

				break


			elif ds2 != 0:

				#rotate left
				vel = 0.5
				print("Rotate left")
				sim.setJointTargetVelocity(jointHandle_l, -vel)
				sim.setJointTargetVelocity(jointHandle_r, vel)
				time.sleep(1.8)
				k = 1

			else :

				#rotate right
				vel = 0.5
				print("Rotate right")
				sim.setJointTargetVelocity(jointHandle_l, vel)
				sim.setJointTargetVelocity(jointHandle_r, -vel)
				time.sleep(2.5)
				k = 1

		elif k == 1 :

			if S3 >= 79.5 and S3 <= 85 :

				vel = 0
				print("Align left")
				sim.setJointTargetVelocity(jointHandle_l, vel)
				sim.setJointTargetVelocity(jointHandle_r, vel)

				k = 0

			elif S3 <= 79.5 :

				vel = 0.5
				print("Align left")
				sim.setJointTargetVelocity(jointHandle_l, -vel)
				sim.setJointTargetVelocity(jointHandle_r, vel)
				k = 1

			elif S3 >= 85 :

				v = 0.5
				print("Align right")
				sim.setJointTargetVelocity(jointHandle_l, vel)
				sim.setJointTargetVelocity(jointHandle_r, -vel)
				k = 1

		else :

			vel = 5.5

			if ds3 == 0 :

				print("Move straight")
				sim.setJointTargetVelocity(jointHandle_l, vel)
				sim.setJointTargetVelocity(jointHandle_r, vel)
				k = 0

			if S3 >= 79.5 and S3 <= 85 :

				print("Move straight")
				sim.setJointTargetVelocity(jointHandle_l, vel)
				sim.setJointTargetVelocity(jointHandle_r, vel)
				k = 0

			elif S3 <= 79.5 :

				v = 1
				print("Align left")
				sim.setJointTargetVelocity(jointHandle_l, 7)
				sim.setJointTargetVelocity(jointHandle_r, 7 + v)
				k = 0

			elif S3 >= 85 :

				v = 1
				print("Align right")
				sim.setJointTargetVelocity(jointHandle_l, 7 + v)
				sim.setJointTargetVelocity(jointHandle_r, 7)
				k = 0




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
