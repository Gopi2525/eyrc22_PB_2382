'''
*****************************************************************************************
*
*        =================================================
*             Pharma Bot Theme (eYRC 2022-23)
*        =================================================
*
*  This script is intended for implementation of Task 4A
*  of Pharma Bot (PB) Theme (eYRC 2022-23).
*
*  Filename:			task_4a.py
*  Created:
*  Last Modified:		02/01/2023
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
# Filename:			task_4a.py
# Functions:		[ Comma separated list of functions in this file ]
# 					
####################### IMPORT MODULES #######################
## You are not allowed to make any changes in this section. ##
##############################################################
import numpy as np
import cv2
from zmqRemoteApi import RemoteAPIClient
import zmq
import os
import time
##############################################################

################# ADD UTILITY FUNCTIONS HERE #################

def homogeneous_transform(xp,yp):
    
    xcg = -0.89
    ycg = -0.89

    T = [[1, 0, 0, xcg],[0, 1, 1, ycg],[0, 0, 0, 0],[0, 0, 0, 1]]
    pb = [[xp], [yp], [0], [1]]

    T = np.array(T,ndmin=4)
    pb = np.array(pb,ndmin=4)

    result=np.matmul(T,pb) 

    xr = result[0][0][0][0]
    yr = -result[0][0][1][0]

    return [xr,yr]

def coordinates_of_signal(node):

    k = (ord(node[0])-5)%10
    l = int(node[1])-1
    x = 0.356*k
    y = 0.356*l

    coordinates = homogeneous_transform(x,y)

    return coordinates

def coordinates_of_barricade(position):

    p1 = position[0:2]
    p2 = position[3:]

    c1 = coordinates_of_signal(p1)
    c2 = coordinates_of_signal(p2)

    return [((c1[0]+c2[0])/2),((c1[1]+c2[1])/2)]

def remove_items(test_list, item):

    res = [i for i in test_list if i != item]
    return res

def shop_ref(num):
    
    a = str(chr(64+num))+"2"
    return a

def pos_of_package(shop_ref,package_num):

    x,y = coordinates_of_signal(shop_ref)
    x = (x+0.04+0.02)+(0.08*package_num)
    y = (y+0.135)

    return [x,y]

def three_d_shape(shape):

    if shape == "Circle":
        return "_cylinder"

    if shape == "Triangle":
        return "_cone"

    if shape == "Square":
        return "_cube"

##############################################################

def place_packages(medicine_package_details, sim, all_models):
    """
	Purpose:
	---
	This function takes details (colour, shape and shop) of the packages present in 
    the arena (using "detect_arena_parameters" function from task_1a.py) and places
    them on the virtual arena. The packages should be inserted only into the 
    designated areas in each shop as mentioned in the Task document.

    Functions from Regular API References should be used to set the position of the 
    packages.

	Input Arguments:
	---
	`medicine_package_details` :	[ list ]
                                nested list containing details of the medicine packages present.
                                Each element of this list will contain 
                                - Shop number as Shop_n
                                - Color of the package as a string
                                - Shape of the package as a string
                                - Centroid co-ordinates of the package			

    `sim` : [ object ]
            ZeroMQ RemoteAPI object

    `all_models` : [ list ]
            list containing handles of all the models imported into the scene
	Returns:

    `all_models` : [ list ]
            list containing handles of all the models imported into the scene
	
	Example call:
	---
	all_models = place_packages(medicine_package_details, sim, all_models)
	"""
    models_directory = os.getcwd()
    packages_models_directory = os.path.join(models_directory, "package_models")
    arena = sim.getObject('/Arena')    
####################### ADD YOUR CODE HERE #########################

    for i in medicine_package_details :

        if i == []:
            continue

        j = 0
        shop = int(i[j][-1])
        o = i[j]
        i=remove_items(i,(i[j]))
        shop_reference = shop_ref(shop)
        paths=[]

        for j in range(0,len(i),3):

            colour = i[j]
            shape = i[j+1]
            model = colour+three_d_shape(shape)+".ttm"

            model_path = packages_models_directory+"/"+model
            paths.append(model_path)

        for l in range(0,len(paths)):

            position_of_package  = (pos_of_package(shop_reference,l)[0],pos_of_package(shop_reference,l)[1],0.02)
            #print(paths[l])
            objectHandle_med_package = sim.loadModel(paths[l])
            sim.setObjectAlias(objectHandle_med_package,o+"_"+"package_"+str(l))
            sim.setObjectParent(objectHandle_med_package,arena,1)
            sim.setObjectPosition(objectHandle_med_package,sim.handle_parent, position_of_package)
            all_models.append(objectHandle_med_package)


####################################################################
    return all_models

def place_traffic_signals(traffic_signals, sim, all_models):
    """
	Purpose:
	---
	This function takes position of the traffic signals present in 
    the arena (using "detect_arena_parameters" function from task_1a.py) and places
    them on the virtual arena. The signal should be inserted at a particular node.

    Functions from Regular API References should be used to set the position of the 
    signals.

	Input Arguments:
	---
	`traffic_signals` : [ list ]
			list containing nodes in which traffic signals are present

    `sim` : [ object ]
            ZeroMQ RemoteAPI object

    `all_models` : [ list ]
            list containing handles of all the models imported into the scene
	Returns:

    `all_models` : [ list ]
            list containing handles of all the models imported into the scene
	None
	
	Example call:
	---
	all_models = place_traffic_signals(traffic_signals, sim, all_models)
	"""
    models_directory = os.getcwd()
    traffic_sig_model = os.path.join(models_directory, "signals", "traffic_signal.ttm" )
    arena = sim.getObject('/Arena')   
####################### ADD YOUR CODE HERE #########################

    n = 1

    for i in traffic_signals:

        position_of_traffic_signal = (coordinates_of_signal(i)[0],coordinates_of_signal(i)[1],0.3)
        objectHandle_traffic_signal = sim.loadModel(traffic_sig_model)
        sim.setObjectAlias(objectHandle_traffic_signal,"traffic_signal_"+str(n))
        sim.setObjectParent(objectHandle_traffic_signal,arena,1)
        sim.setObjectPosition(objectHandle_traffic_signal,sim.handle_parent, position_of_traffic_signal)
        all_models.append(objectHandle_traffic_signal)

        n = n+1
    
####################################################################
    return all_models

def place_start_end_nodes(start_node, end_node, sim, all_models):
    """
	Purpose:
	---
	This function takes position of start and end nodes present in 
    the arena and places them on the virtual arena. 
    The models should be inserted at a particular node.

    Functions from Regular API References should be used to set the position of the 
    start and end nodes.

	Input Arguments:
	---
	`start_node` : [ string ]
    `end_node` : [ string ]
					

    `sim` : [ object ]
            ZeroMQ RemoteAPI object

    `all_models` : [ list ]
            list containing handles of all the models imported into the scene
	Returns:

    `all_models` : [ list ]
            list containing handles of all the models imported into the scene
	---
	None
	
	Example call:
	---
	all_models = place_start_end_nodes(start_node, end_node, sim, all_models)
	"""
    models_directory = os.getcwd()
    start_node_model = os.path.join(models_directory, "signals", "start_node.ttm" )
    end_node_model = os.path.join(models_directory, "signals", "end_node.ttm" )
    arena = sim.getObject('/Arena')   
####################### ADD YOUR CODE HERE #########################

    position_of_start_signal = (coordinates_of_signal(start_node)[0],coordinates_of_signal(start_node)[1],0.3)
    position_of_end_signal = (coordinates_of_signal(end_node)[0],coordinates_of_signal(end_node)[1],0.3)

    objectHandle_start = sim.loadModel(start_node_model)
    objectHandle_end = sim.loadModel(end_node_model)

    sim.setObjectAlias(objectHandle_start,"start_signal")
    sim.setObjectAlias(objectHandle_end,"end_signal")

    sim.setObjectParent(objectHandle_start,arena,1)
    sim.setObjectParent(objectHandle_end,arena,1)

    sim.setObjectPosition(objectHandle_start,sim.handle_parent, position_of_start_signal)
    sim.setObjectPosition(objectHandle_end,sim.handle_parent, position_of_end_signal)

    all_models.append(objectHandle_start)
    all_models.append(objectHandle_end)

####################################################################
    return all_models

def place_horizontal_barricade(horizontal_roads_under_construction, sim, all_models):
    """
	Purpose:
	---
	This function takes the list of missing horizontal roads present in 
    the arena (using "detect_arena_parameters" function from task_1a.py) and places
    horizontal barricades on virtual arena. The barricade should be inserted 
    between two nodes as shown in Task document.

    Functions from Regular API References should be used to set the position of the 
    horizontal barricades.

	Input Arguments:
	---
	`horizontal_roads_under_construction` : [ list ]
			list containing missing horizontal links		

    `sim` : [ object ]
            ZeroMQ RemoteAPI object

    `all_models` : [ list ]
            list containing handles of all the models imported into the scene
	Returns:

    `all_models` : [ list ]
            list containing handles of all the models imported into the scene
	---
	None
	
	Example call:
	---
	all_models = place_horizontal_barricade(horizontal_roads_under_construction, sim, all_models)
	"""
    models_directory = os.getcwd()
    horiz_barricade_model = os.path.join(models_directory, "barricades", "horizontal_barricade.ttm" )
    arena = sim.getObject('/Arena')  
####################### ADD YOUR CODE HERE #########################

    n = 1

    for i in horizontal_roads_under_construction:

        position_of_horizontal_baricade = (coordinates_of_barricade(i)[0],coordinates_of_barricade(i)[1],0.025)
        objectHandle_horizontal_baricade = sim.loadModel(horiz_barricade_model)
        sim.setObjectAlias(objectHandle_horizontal_baricade,"horizontal_baricade_"+str(n))
        sim.setObjectParent(objectHandle_horizontal_baricade,arena,1)
        sim.setObjectPosition(objectHandle_horizontal_baricade,sim.handle_parent, position_of_horizontal_baricade)
        all_models.append(objectHandle_horizontal_baricade)

        n = n+1
    
####################################################################
    return all_models


def place_vertical_barricade(vertical_roads_under_construction, sim, all_models):
    """
	Purpose:
	---
	This function takes the list of missing vertical roads present in 
    the arena (using "detect_arena_parameters" function from task_1a.py) and places
    vertical barricades on virtual arena. The barricade should be inserted 
    between two nodes as shown in Task document.

    Functions from Regular API References should be used to set the position of the 
    vertical barricades.

	Input Arguments:
	---
	`vertical_roads_under_construction` : [ list ]
			list containing missing vertical links		

    `sim` : [ object ]
            ZeroMQ RemoteAPI object

    `all_models` : [ list ]
            list containing handles of all the models imported into the scene
	Returns:

    `all_models` : [ list ]
            list containing handles of all the models imported into the scene
	---
	None
	
	Example call:
	---
	all_models = place_vertical_barricade(vertical_roads_under_construction, sim, all_models)
	"""
    models_directory = os.getcwd()
    vert_barricade_model = os.path.join(models_directory, "barricades", "vertical_barricade.ttm" )
    arena = sim.getObject('/Arena') 
####################### ADD YOUR CODE HERE #########################

    n = 1

    for i in vertical_roads_under_construction:

        position_of_vertical_baricade = (coordinates_of_barricade(i)[0],coordinates_of_barricade(i)[1],0.025)
        objectHandle_vertical_baricade = sim.loadModel(vert_barricade_model)
        sim.setObjectAlias(objectHandle_vertical_baricade,"vertical_baricade_"+str(n))
        sim.setObjectParent(objectHandle_vertical_baricade,arena,1)
        sim.setObjectPosition(objectHandle_vertical_baricade,sim.handle_parent, position_of_vertical_baricade)
        all_models.append(objectHandle_vertical_baricade)

        n = n+1
    
####################################################################
    return all_models

if __name__ == "__main__":
    client = RemoteAPIClient()
    sim = client.getObject('sim')

    # path directory of images in test_images folder
    img_dir = os.getcwd() + "/test_imgs/"

    i = 0
    config_img = cv2.imread(img_dir + 'maze_' + str(i) + '.png')

    print('\n============================================')
    print('\nFor maze_0.png')

    # object handles of each model that gets imported to the scene can be stored in this list
    # at the end of each test image, all the models will be removed    
    all_models = []

    # import task_1a.py. Make sure that task_1a.py is in same folder as task_4a.py
    task_1 = __import__('task_1a')
    detected_arena_parameters = task_1.detect_arena_parameters(config_img)

    # obtain required arena parameters
    medicine_package_details = detected_arena_parameters["medicine_packages"]
    traffic_signals = detected_arena_parameters['traffic_signals']
    start_node = detected_arena_parameters['start_node']
    end_node = detected_arena_parameters['end_node']
    horizontal_roads_under_construction = detected_arena_parameters['horizontal_roads_under_construction']
    vertical_roads_under_construction = detected_arena_parameters['vertical_roads_under_construction'] 

    print("[1] Setting up the scene in CoppeliaSim")
    all_models = place_packages(medicine_package_details, sim, all_models)
    all_models = place_traffic_signals(traffic_signals, sim, all_models)
    all_models = place_horizontal_barricade(horizontal_roads_under_construction, sim, all_models)
    all_models = place_vertical_barricade(vertical_roads_under_construction, sim, all_models)
    all_models = place_start_end_nodes(start_node, end_node, sim, all_models)
    print("[2] Completed setting up the scene in CoppeliaSim")

    # wait for 10 seconds and then remove models
    time.sleep(10)
    print("[3] Removing models for maze_0.png")

    for i in all_models:
        sim.removeModel(i)

   
    choice = input('\nDo you want to run your script on all test images ? => "y" or "n": ')
    
    if choice == 'y':
        for i in range(1,5):

            print('\n============================================')
            print('\nFor maze_' + str(i) +'.png')
            config_img = cv2.imread(img_dir + 'maze_' + str(i) + '.png')

            # object handles of each model that gets imported to the scene can be stored in this list
            # at the end of each test image, all the models will be removed    
            all_models = []

            # import task_1a.py. Make sure that task_1a.py is in same folder as task_4a.py
            task_1 = __import__('task_1a')
            detected_arena_parameters = task_1.detect_arena_parameters(config_img)

            # obtain required arena parameters
            medicine_package_details = detected_arena_parameters["medicine_packages"]
            traffic_signals = detected_arena_parameters['traffic_signals']
            start_node = detected_arena_parameters['start_node']
            end_node = detected_arena_parameters['end_node']
            horizontal_roads_under_construction = detected_arena_parameters['horizontal_roads_under_construction']
            vertical_roads_under_construction = detected_arena_parameters['vertical_roads_under_construction'] 

            print("[1] Setting up the scene in CoppeliaSim")
            place_packages(medicine_package_details, sim, all_models)
            place_traffic_signals(traffic_signals, sim, all_models)
            place_horizontal_barricade(horizontal_roads_under_construction, sim, all_models)
            place_vertical_barricade(vertical_roads_under_construction, sim, all_models)
            place_start_end_nodes(start_node, end_node, sim, all_models)
            print("[2] Completed setting up the scene in CoppeliaSim")

            # wait for 10 seconds and then remove models
            time.sleep(10)
            print("[3] Removing models for maze_" + str(i) + '.png')
            for i in all_models:
                sim.removeModel(i)
            