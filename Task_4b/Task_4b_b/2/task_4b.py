'''
*****************************************************************************************
*
*        		===============================================
*           		Pharma Bot (PB) Theme (eYRC 2022-23)
*        		===============================================
*
*  This script is to implement Task 4B of Pharma Bot (PB) Theme (eYRC 2022-23).
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
# Filename:			task_4b.py
# Functions:
# 					[ Comma separated list of functions in this file ]

####################### IMPORT MODULES #######################
## You are not allowed to make any changes in this section. ##
## You have to implement this task with the three available ##
## modules for this task (numpy, opencv)                    ##
##############################################################
import socket
import time
import os, sys
from zmqRemoteApi import RemoteAPIClient
import traceback
import zmq
import numpy as np
import cv2
from pyzbar.pyzbar import decode
import json
import random
from threading import Thread,Event
import math
##############################################################

class Storage_log():

    def past_data():

        Storage_log.data = []

def und_undwrm(img):

    h = 316
    w = 467
    cameraMatrix = np.array( [[432.64843938,0.,323.04192955],[0.,431.83992831,245.06752113],[0.,0.,1.]])
    newCameraMatrix = np.array([[239.79382324,0.,342.27052461],[0.,245.33255005,268.8847273 ],[0.,0.,1.]])
    roi = (104, 105, 467, 316)
    dist = np.array([[-0.40167891,0.19897568,0.00285976,0.00067859,-0.04539993]])

    # Undistort
    dst = cv2.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    # Undistort with Remapping
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
    dst1 = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    dst1 = dst[y:y+h, x:x+w]

    return dst,dst1

def centroid(vertexes):


     _x_list = [vertex [0] for vertex in vertexes]
     _y_list = [vertex [1] for vertex in vertexes]
     _len = len(vertexes)
     _x = sum(_x_list) / _len
     _y = sum(_y_list) / _len

     return(int(_x), int(_y))

def checkKey(dic, key):

    if key in dic.keys():

        #Present

        return 1

    else:

        #Not present

        return 0

class Storage():

    def dictonary_past_data():

        Storage.data_1 = []
        Storage.data_2 = []
        Storage.data_3 = []
        Storage.data_4 = []
        Storage.data_5 = []
        Storage.data = []
        Storage.data_corners = []

#####################################################################################

def perspective_transform(image):

    """
    Purpose:
    ---
    This function takes the image as an argument and returns the image after
    applying perspective transform on it. Using this function, you should
    crop out the arena from the full frame you are receiving from the
    overhead camera feed.

    HINT:
    Use the ArUco markers placed on four corner points of the arena in order
    to crop out the required portion of the image.

    Input Arguments:
    ---
    `image` :	[ np array ]
            np array of image returned by cv2 library

    Returns:
    ---
    `warped_image` : [ np array ]
            return cropped arena image as a np array

    Example call:
    ---
    warped_image = perspective_transform(image)
    """
    warped_image = []
#################################  ADD YOUR CODE HERE  ###############################

    img_sent = image.copy()
    ArUco_details_dict, ArUco_corners = pb_theme.detect_ArUco_details(img_sent)

    img  = image

    img_copy = np.copy(img)
    img_copy = cv2.cvtColor(img_copy,cv2.COLOR_BGR2RGB)

    ##### Add data to storage individual channels  & Update dictonary #####
    for key in range(1,5):

        o = checkKey(ArUco_details_dict, key)
        if o == 1:
            if key == 1 :
                Storage.data_1.append([ArUco_details_dict[key],ArUco_corners[key]])
            elif key == 2 :
                Storage.data_2.append([ArUco_details_dict[key],ArUco_corners[key]])
            elif key == 3 :
                Storage.data_3.append([ArUco_details_dict[key],ArUco_corners[key]])
            elif key == 4 :
                Storage.data_4.append([ArUco_details_dict[key],ArUco_corners[key]])
        else:

            if key == 1 :
                ArUco_details_dict[key] = Storage.data_1[-1][0]
                ArUco_corners[key] = Storage.data_1[-1][1]
            elif key == 2 :
                ArUco_details_dict[key] = Storage.data_2[-1][0]
                ArUco_corners[key] = Storage.data_2[-1][1]
            elif key == 3 :
                ArUco_details_dict[key] = Storage.data_3[-1][0]
                ArUco_corners[key] = Storage.data_3[-1][1]
            elif key == 4 :
                ArUco_details_dict[key] = Storage.data_4[-1][0]
                ArUco_corners[key] = Storage.data_4[-1][1]

    pt_A = [ArUco_details_dict[1][0][0], ArUco_details_dict[1][0][1]]
    pt_B = [ArUco_details_dict[2][0][0], ArUco_details_dict[2][0][1]]
    pt_C = [ArUco_details_dict[3][0][0], ArUco_details_dict[3][0][1]]
    pt_D = [ArUco_details_dict[4][0][0], ArUco_details_dict[4][0][1]]

    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0],[0, maxHeight - 1],[maxWidth - 1, maxHeight - 1],[maxWidth - 1, 0]])

    # Compute the perspective transform M
    M = cv2.getPerspectiveTransform(input_pts,output_pts)

    out = cv2.warpPerspective(img,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)
    out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
    out = cv2.flip(out, 0)

    warped_image = out

######################################################################################

    return warped_image




def transform_values(image):

    """
    Purpose:
    ---
    This function takes the image as an argument and returns the
    position and orientation of the ArUco marker (with id 5), in the
    CoppeliaSim scene.

    Input Arguments:
    ---
    `image` :	[ np array ]
            np array of image returned by camera

    Returns:
    ---
    `scene_parameters` : [ list ]
            a list containing the position and orientation of ArUco 5
            scene_parameters = [c_x, c_y, c_angle] where
            c_x is the transformed x co-ordinate [float]
            c_y is the transformed y co-ordinate [float]
            c_angle is the transformed angle [angle]

    HINT:
        Initially the image should be cropped using perspective transform
        and then values of ArUco (5) should be transformed to CoppeliaSim
        scale.

    Example call:
    ---
    scene_parameters = transform_values(image)
    """
    scene_parameters = []
#################################  ADD YOUR CODE HERE  ###############################

    image_sended = image.copy()
    ArUco_details_dict, ArUco_corners = pb_theme.detect_ArUco_details(image_sended)
    o = checkKey(ArUco_details_dict, 5)

    if o == 1:
        Storage.data_5.append([ArUco_details_dict[5],ArUco_corners[5]])

    else:
        ArUco_details_dict[5] = Storage.data_5[-1][0]


    x5y5 = ArUco_details_dict[5][0]

    xcg = -image.shape[0]/2
    ycg = -image.shape[1]/2

    T = [[1, 0, 0, xcg],[0, 1, 1, ycg],[0, 0, 0, 0],[0, 0, 0, 1]]
    pb = [[x5y5[0]], [x5y5[1]], [0], [1]]

    T = np.array(T,ndmin=4)
    pb = np.array(pb,ndmin=4)

    result=np.matmul(T,pb)

    xgr = result[0][0][0][0]
    ygr = result[0][0][1][0]


    xgr = (204/image.shape[0])*xgr
    ygr = (204/image.shape[1])*ygr

    scene_parameters = [xgr/100,ygr/100,ArUco_details_dict[5][1]]
    #print(scene_parameters)

######################################################################################

    return scene_parameters



def set_values_i(scene_parameters):
    """
    Purpose:
    ---
    This function takes the scene_parameters, i.e. the transformed values for
    position and orientation of the ArUco marker, and sets the position and
    orientation in the CoppeliaSim scene.

    Input Arguments:
    ---
    `scene_parameters` :	[ list ]
            list of co-ordinates and orientation obtained from transform_values()
            function

    Returns:
    ---
    None

    HINT:
        Refer Regular API References of CoppeliaSim to find out functions that can
        set the position and orientation of an object.

    Example call:
    ---
    set_values(scene_parameters)
    """
    #aruco_handle = sim.getObject('/aruco_5')
#################################  ADD YOUR CODE HERE  ###############################

    alphabot_handle = sim.getObject('/Alphabot')

    #print(scene_parameters)

    position_robot = (-scene_parameters[0]-0.025,scene_parameters[1]-0.025,+0.030)
    eulerAngles_robot = [1.5708,math.radians(scene_parameters[2]),+1.5708]
    print(eulerAngles_robot)
    #eulerAngles_robot = [0.0, 3.141592502593994,-((scene_parameters[2]*(3.141592502593994/180))-3.141592502593994)]

    sim.setObjectPosition(alphabot_handle,sim.handle_parent, position_robot)
    sim.setObjectOrientation(alphabot_handle,sim.handle_parent,eulerAngles_robot)


######################################################################################

    return None



class Graph:

    def __init__(self, adjacency_list):
        self.adjacency_list = adjacency_list

    def get_neighbors(self, v):
        return self.adjacency_list[v]

    def h(self, n):
        H = {'A1': 1, 'A2': 1, 'A3': 1, 'A4': 1, 'A5': 1, 'A6': 1, 'B1': 1, 'B2': 1, 'B3': 1, 'B4': 1, 'B5': 1, 'B6': 1, 'C1': 1, 'C2': 1, 'C3': 1, 'C4': 1, 'C5': 1, 'C6': 1, 'D1': 1, 'D2': 1, 'D3': 1, 'D4': 1, 'D5': 1, 'D6': 1, 'E1': 1, 'E2': 1, 'E3': 1, 'E4': 1, 'E5': 1, 'E6': 1, 'F1': 1, 'F2': 1, 'F3': 1, 'F4': 1, 'F5': 1, 'F6': 1}

        return H[n]

    def a_star_algorithm(self, start_node, stop_node):

        open_list = set([start_node])
        closed_list = set([])

        g = {}

        g[start_node] = 0

        parents = {}
        parents[start_node] = start_node

        while len(open_list) > 0:
            n = None

            for v in open_list:
                if n == None or g[v] + self.h(v) < g[n] + self.h(n):
                    n = v;

            if n == None:
                print('Path does not exist!')
                return None

            if n == stop_node:
                reconst_path = []

                while parents[n] != n:
                    reconst_path.append(n)
                    n = parents[n]

                reconst_path.append(start_node)

                reconst_path.reverse()

                #print('Path found: {}'.format(reconst_path))
                return reconst_path

            for (m, weight) in self.get_neighbors(n):

                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight

                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        parents[m] = n

                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)


            open_list.remove(n)
            closed_list.add(n)

        print('Path does not exist!')
        return None

def detect_vertical_roads_under_construction(maze_image):

 dict={}
 vertical_roads_under_construction = []
 part=[];x=94;y=106;x1=107;y1=193;vertical_roads_under_construction=[]
 image = maze_image
 for j in range(1,6):
  x=94;y=106
  for i  in range(65,71):
   crop =image[x1:y1,x:y]
   if np.count_nonzero(crop)!=0:
     one=chr(i)+str(j)
     two=chr(i)+str(j+1)
     vertical_roads_under_construction.append(one+two)
     vertical_roads_under_construction.append(two+one)
   x=x+100;y=y+100
  x1=x1+100;y1=y1+100
 return vertical_roads_under_construction

def detect_horizontal_roads_under_construction(maze_image):

 horizontal_roads_under_construction = []
 x=107;y=193;x1=94;y1=106;horizontal_roads_under_construction=[]
 image = maze_image
 for j in range(1,7):
  x=107;y=193
  for i  in range(65,70):
   crop =image[x1:y1,x:y]
   if np.count_nonzero(crop)!=0:
     one=chr(i)+str(j)
     two=chr(i+1)+str(j)
     horizontal_roads_under_construction.append(one+two)
     horizontal_roads_under_construction.append(two+one)
   x=x+100;y=y+100
  x1=x1+100;y1=y1+100

 return horizontal_roads_under_construction

def pathdic(a):
 dict={}
 for i in range(65,71):
   for j in range(1,7):
    p=chr(i)+str(j)
    dict[p]={}
    if(i-1>=65):
     one=chr(i-1)+str(j)
     if (p+one not in a):
      dict[p][one]=1
    if(i+1<=70):
     two=chr(i+1)+str(j)
     if (p+two not in a):
      dict[p][two]=1
    if(j-1>=1):
     three=chr(i)+str(j-1)
     if (p+three not in a):
      dict[p][three]=1
    if(j+1<=6):
     four=chr(i)+str(j+1)
     if (p+four not in a):
      dict[p][four]=1


 return dict

def converter(dictonary_old):

    adjacency_list = {}
    all_keys  = list(dictonary_old.keys())
    for key in all_keys:
        key_in = list(dictonary_old[key].keys())
        t = []
        for k in key_in:
            p = dictonary_old[key][k]
            if p == 0:
                continue
            t.append((k,p))
        if len(t)>0:
            adjacency_list[key]=t

    return adjacency_list

def statefinder(x,y):

	###statefinder(current,future)

	if ord(x[0]) < ord(y[0]): ### alpha inc
		c = 4
	if ord(x[0]) > ord(y[0]): ### alpha dec
		c = 2
	if int(x[1]) < int(y[1]): ### num inc
		c = 3
	if int(x[1]) > int(y[1]): ### num dec
		c = 1

	return c

def state_to_moves(past,future):

	p = past
	f = future

	if p == 1 and f == 4 :
		s = 1
	elif p == 4 and f == 1 :
		s = 2
	elif p == f :
		s = 0
	elif p<f :
		s = 2
	elif p>f :
		s = 1

	return s

##############################################################

def detect_all_nodes(image):
	"""
	Purpose:
	---
	This function takes the image as an argument and returns a list of
	nodes in which traffic signals, start_node and end_node are present in the image

	Input Arguments:
	---
	`maze_image` :	[ np array ]
			np array of image returned by cv2 library
	Returns:
	---
	`traffic_signals, start_node, end_node` : [ list ], str, str
			list containing nodes in which traffic signals are present, start and end node too

	Example call:
	---
	traffic_signals, start_node, end_node = detect_all_nodes(maze_image)
	"""

	##############	ADD YOUR CODE HERE	##############

	x=94;y=106;x1=94;y1=106
	image = image
	for f in range(65,71):
		x=94;y=106
		for i  in range(1,7):
			crop=image[x:y,x1:y1]
			x=x+100;y=y+100
			lower_bound = np.array([0,250,0])
			upper_bound = np.array([0,255,0])
			imagemask = cv2.inRange(crop, lower_bound, upper_bound)
			if np.count_nonzero(imagemask)!=0:
				start_node=(chr(f)+str(i))
			lower_bound = np.array([180,40,100])
			upper_bound = np.array([189,43,105])
			imagemask = cv2.inRange(crop, lower_bound, upper_bound)
			if np.count_nonzero(imagemask)!=0:
				end_node=(chr(f)+str(i))

		x1=x1+100;y1=y1+100

	traffic_signals = [];x=94;y=106;x1=94;y1=106
	#image = maze_image
	for f in range(65,71):
		x=94;y=106
		for i  in range(1,7):
			crop=image[x:y,x1:y1]
			x=x+100;y=y+100
			lower_bound = np.array([0,0,250])
			upper_bound = np.array([0,0,255])
			imagemask = cv2.inRange(crop, lower_bound, upper_bound)
			if np.count_nonzero(imagemask)!=0:
				traffic_signals.append(chr(f)+str(i))
		x1=x1+100;y1=y1+100

	##################################################

	return start_node,end_node,traffic_signals

def detect_paths_to_graph(image):
	"""
	Purpose:
	---
	This function takes the image as an argument and returns a dictionary of the
	connect path from a node to other nodes and will be used for path planning

	HINT: Check for the road besides the nodes for connectivity

	Input Arguments:
	---
	`maze_image` :	[ np array ]
			np array of image returned by cv2 library
	Returns:
	---
	`paths` : { dictionary }
			Every node's connection to other node and set it's value as edge value
			Eg. : { "D3":{"C3":1, "E3":1, "D2":1, "D4":1},
					"D5":{"C5":1, "D2":1, "D6":1 }  }

			Why edge value 1? -->> since every road is equal

	Example call:
	---
	paths = detect_paths_to_graph(maze_image)
	"""

	paths = {}

	##############	ADD YOUR CODE HERE	##############

	a = detect_horizontal_roads_under_construction(image)
	b = detect_vertical_roads_under_construction(image)
	c = a+b
	paths = pathdic(c)

	##################################################

	return paths

def detect_arena_parameters(maze_image):
	"""
	Purpose:
	---
	This function takes the image as an argument and returns a dictionary
	containing the details of the different arena parameters in that image

	The arena parameters are of four categories:
	i) traffic_signals : list of nodes having a traffic signal
	ii) start_node : Start node which is mark in light green
	iii) end_node : End node which is mark in Purple
	iv) paths : list containing paths

	These four categories constitute the four keys of the dictionary

	Input Arguments:
	---
	`maze_image` :	[ np array ]
			np array of image returned by cv2 library
	Returns:
	---
	`arena_parameters` : { dictionary }
			dictionary containing details of the arena parameters

	Example call:
	---
	arena_parameters = detect_arena_parameters(maze_image)

	Eg. arena_parameters={"traffic_signals":[],
	                      "start_node": "E4",
	                      "end_node":"A3",
	                      "paths": {}}
	"""
	arena_parameters = {}

	##############	ADD YOUR CODE HERE	##############

	a = detect_all_nodes(maze_image)
	paths = detect_paths_to_graph(maze_image)
	arena_parameters={"traffic_signals": a[2],"start_node":a[0], "end_node":a[1], "paths":paths}

	##################################################

	return arena_parameters

def path_planning(graph, start, end):

	"""
	Purpose:
	---
	This function takes the graph(dict), start and end node for planning the shortest path

	** Note: You can use any path planning algorithm for this but need to produce the path in the form of
	list given below **

	Input Arguments:
	---
	`graph` :	{ dictionary }
			dict of all connecting path
	`start` :	str
			name of start node
	`end` :		str
			name of end node


	Returns:
	---
	`backtrace_path` : [ list of nodes ]
			list of nodes, produced using path planning algorithm

		eg.: ['C6', 'C5', 'B5', 'B4', 'B3']

	Example call:
	---
	arena_parameters = detect_arena_parameters(maze_image)
	"""

	backtrace_path=[]

	##############	ADD YOUR CODE HERE	##############

	adjacency_list =  converter(graph)
	graph1 = Graph(adjacency_list)
	backtrace_path = graph1.a_star_algorithm(start, end)

	##################################################


	return backtrace_path

def paths_to_moves(paths, traffic_signal):

	"""
	Purpose:
	---
	This function takes the list of all nodes produces from the path planning algorithm
	and connecting both start and end nodes

	Input Arguments:
	---
	`paths` :	[ list of all nodes ]
			list of all nodes connecting both start and end nodes (SHORTEST PATH)
	`traffic_signal` : [ list of all traffic signals ]
			list of all traffic signals
	---
	`moves` : [ list of moves from start to end nodes ]
			list containing moves for the bot to move from start to end

			Eg. : ['UP', 'LEFT', 'UP', 'UP', 'RIGHT', 'DOWN']

	Example call:
	---
	moves = paths_to_moves(paths, traffic_signal)
	"""

	list_moves=[]

	##############	ADD YOUR CODE HERE	##############

	all_moves = ['STRAIGHT','RIGHT','LEFT','WAIT_5']

	p = 1
	f = statefinder(paths[0],paths[1])
	list_moves.append(all_moves[state_to_moves(p,f)])

	for i in range(1,len(paths)-1):
		p = statefinder(paths[i-1],paths[i])
		f = statefinder(paths[i],paths[i+1])
		for j in traffic_signal:
			if paths[i]==j:
				list_moves.append(all_moves[3])
		list_moves.append(all_moves[state_to_moves(p,f)])
		#print([p,f])


	##################################################

	return list_moves

def shop_nodes(shop):

	if shop == "Shop_1" :
		location = "A2"
	elif shop == "Shop_2" :
		location = "B2"
	elif shop == "Shop_3":
		location = "C2"
	elif shop == "Shop_4" :
		location = "D2"
	elif shop == "Shop_5" :
		location = "E2"
	else:
		location = "INVALID input"

	return location

def shop_with_packages(medicine_package_details):

    shop_data = []
    for i in range(0,len(medicine_package_details)):
        d = ["Shop_"+str(i+1),int(len(medicine_package_details[i])/4)]
        shop_data.append(d)

    return shop_data


def emulation(event,sim):

	#####initialising camera #####
    vid = cv2.VideoCapture(0)

    ##### initializing storage #####
    Storage.dictonary_past_data()
    Storage_log.past_data()

    while(True):

        ret, frame = vid.read()
        dst,dst1 = und_undwrm(frame)

        img_1 = dst.copy()
        frame_1 = frame.copy()

        warped_image = perspective_transform(img_1)
        warped_image = cv2.bilateralFilter(warped_image, 9, 75, 75)
        kernel = np.array([[0, -1, 0],[-1, 5,-1],[0, -1, 0]])
        image_sharp = cv2.filter2D(src=warped_image, ddepth=-1, kernel=kernel)
        warped_image_sharpened = image_sharp.copy()
        scene_parameters = transform_values(warped_image_sharpened)
        set_values_i(scene_parameters)

        #cv2.imshow('img_1',img_1)
        cv2.imshow('warped_image_sharpened',warped_image_sharpened)
        cv2.imshow("Frame",frame)

        if event.is_set():
        	break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()



def place_packages(sim,med_pack,position,operation):

	objectHandle_med_package = sim.getObject('/'+med_pack)
	alphabot_handle = sim.getObject('/Alphabot')
	arena_handle = sim.getObject('/Arena')

	if operation == "PICKED UP":

		position_package = (0.05,0,0)
		eulerAngles_package = [0,1.5708,0]
		print(position_package)

		sim.setObjectParent(objectHandle_med_package,alphabot_handle,1)
		sim.setObjectPosition(objectHandle_med_package,sim.handle_parent, position_package)
		sim.setObjectOrientation(objectHandle_med_package,sim.handle_parent,eulerAngles_package)

	if operation == "DELIVERED":

		xp = (pb_theme.position_data(position)[0])
		yp = -(pb_theme.position_data(position)[1])
		position_package = (xp,yp,0.05)
		eulerAngles_package = [0,0,0]
		print(position_package)
		print(position)

		sim.setObjectParent(objectHandle_med_package,arena_handle,1)
		sim.setObjectPosition(objectHandle_med_package,sim.handle_parent, position_package)
		sim.setObjectOrientation(objectHandle_med_package,sim.handle_parent,eulerAngles_package)

		print(sim.getObjectPosition(objectHandle_med_package,arena_handle))

	return None

## Import PB_theme_functions code
try:
	pb_theme = __import__('PB_theme_functions')

except ImportError:
	print('\n[ERROR] PB_theme_functions.py file is not present in the current directory.')
	print('Your current directory is: ', os.getcwd())
	print('Make sure PB_theme_functions.py is present in this current directory.\n')
	sys.exit()

except Exception as e:
	print('Your PB_theme_functions.py throwed an Exception, kindly debug your code!\n')
	traceback.print_exc(file=sys.stdout)
	sys.exit()

def task_4b_implementation(sim):
	"""
	Purpose:
	---
	This function contains the implementation logic for task 4B

	Input Arguments:
	---
    `sim` : [ object ]
            ZeroMQ RemoteAPI object

	You are free to define additional input arguments for this function.

	Returns:
	---
	You are free to define output parameters for this function.

	Example call:
	---
	task_4b_implementation(sim)
	"""

	##################	ADD YOUR CODE HERE	##################

	
	arena_parameters = detect_arena_parameters(config_img)
	new_medicine_package_details = medicine_package_details[:]
	current_position = arena_parameters["start_node"]

	### shops with medicine packages ###
	details = shop_with_packages(new_medicine_package_details)

	### loop with shop data ###
	for i in range(0,len(details)):

		if details[i][1] != 0:

			number_of_traversals = (details[i][1]*2)+1
			delivery = 0
			for j in range(0,number_of_traversals):

				img = cv2.imread("qr_drop_5.png")
				Qr_codes_details = pb_theme.detect_Qr_details_drop(img)
				#print(Qr_codes_details)

				traversal_status = ["continue"]
				colour_for_rgb_pi = "Turn off"
				if j == 0:
					target = shop_nodes(new_medicine_package_details[i][0])
					print("Going to shop")
				if j == number_of_traversals-1:
					print("Going to end node")
					target = arena_parameters["end_node"]
					traversal_status = ["final traversal"]
				if  j == number_of_traversals:
					break
				if j != 0 and j != number_of_traversals-1 :
					if  j%2 == 0:
						### even -- to shop path
						print("Going to shop")
						med_pack = Qr_codes_details[delivery][0]
						target = shop_nodes(new_medicine_package_details[i][0])
					else:
						### odd --- delivery path
						print("Going to deliver at : "+str(Qr_codes_details[delivery][1]))
						med_pack = Qr_codes_details[delivery][0]
						target = Qr_codes_details[delivery][1]
						colour_for_rgb_pi = colours_for_rgb_all[delivery]
						shape_for_act_log = shapes_all[delivery]
						delivery = delivery+1


				### communication ###
				print(str(current_position)+" >>> "+str(target))
				colour = ["colour",colour_for_rgb_pi]
				pb_theme.send_message_via_socket(connection_2,pb_theme.json.dumps(colour))
				print(colour)
				time.sleep(1)
				pb_theme.send_message_via_socket(connection_2,pb_theme.json.dumps(traversal_status))
				#print(traversal_status)
				time.sleep(1)
				back_path=path_planning(arena_parameters["paths"], current_position,target)
				moves=paths_to_moves(back_path,arena_parameters["traffic_signals"])
				#print(back_path)
				#print(moves)
				back_path.insert(0,"paths")
				moves.insert(0,"moves")
				pb_theme.send_message_via_socket(connection_2,pb_theme.json.dumps(back_path))
				#print("sent path")
				time.sleep(1)
				pb_theme.send_message_via_socket(connection_2,pb_theme.json.dumps(moves))
				#print("sent moves")
				end_of_path = 1

				### activity log ###

				while True:
					message = pb_theme.receive_message_via_socket(connection_2)
					if message:
						message = json.loads(message)
						dats = message.copy()
						dats.append(med_pack)
						file1 = open("MyFile1.txt","a")
						file1.truncate(0)
						L = str(dats)
						file1.writelines(L)
						file1.close()
						time.sleep(0.5)
						if type(message) == list:

							if message[5] == "DELIVERED" :
								#place_packages(sim,med_pack,message[0],message[5])
								activity_log_message_package = str(message[5])+": "+colour_for_rgb_pi+","+ shape_for_act_log.capitalize() +" AT "+str(message[0])
								pb_theme.send_message_via_socket(connection_1,activity_log_message_package)
								print(activity_log_message_package)
							if message[5] == "PICKED UP" :
								#place_packages(sim,med_pack,message[0],message[5])
								activity_log_message_package = str(message[5])+": "+colour_for_rgb_pi+","+ shape_for_act_log.capitalize() +","+str(message[0])
								pb_theme.send_message_via_socket(connection_1,activity_log_message_package)
								print(activity_log_message_package)
							if any(end_of_path == item for item in message):
								#print("Code executed")
								print("########################################################")
								break

				if j == 0 :
					print(Qr_codes_details)
					colours_for_rgb_all = pb_theme.colours_rgb(Qr_codes_details)
					shapes_all = pb_theme.shapes_rgb(Qr_codes_details)
				current_position = message[0]



	##########################################################


if __name__ == "__main__":

	host = ''
	port = 5050


	## Connect to CoppeliaSim arena
	coppelia_client = RemoteAPIClient()
	sim = coppelia_client.getObject('sim')

	## Set up new socket server
	try:
		server = pb_theme.setup_server(host, port)
		print("Socket Server successfully created")

		# print(type(server))

	except socket.error as error:
		print("Error in setting up server")
		print(error)
		sys.exit()


	## Set up new connection with a socket client (PB_task3d_socket.exe)
	try:
		print("\nPlease run PB_socket.exe program to connect to PB_socket client")
		connection_1, address_1 = pb_theme.setup_connection(server)
		print("Connected to: " + address_1[0] + ":" + str(address_1[1]))

	except KeyboardInterrupt:
		sys.exit()

	# ## Set up new connection with Raspberry Pi
	try:
		print("\nPlease connect to Raspberry pi client")
		connection_2, address_2 = pb_theme.setup_connection(server)
		print("Connected to: " + address_2[0] + ":" + str(address_2[1]))

	except KeyboardInterrupt:
		sys.exit()

	## Send setup message to PB_socket
	pb_theme.send_message_via_socket(connection_1, "SETUP")

	message = pb_theme.receive_message_via_socket(connection_1)
	## Loop infinitely until SETUP_DONE message is received
	while True:
		if message == "SETUP_DONE":
			break
		else:
			print("Cannot proceed further until SETUP command is received")
			message = pb_theme.receive_message_via_socket(connection_1)

	try:
		# obtain required arena parameters
		config_img = cv2.imread("config_image.png")
		detected_arena_parameters = pb_theme.detect_arena_parameters(config_img)
		medicine_package_details = detected_arena_parameters["medicine_packages"]
		traffic_signals = detected_arena_parameters['traffic_signals']
		start_node = detected_arena_parameters['start_node']
		end_node = detected_arena_parameters['end_node']
		horizontal_roads_under_construction = detected_arena_parameters['horizontal_roads_under_construction']
		vertical_roads_under_construction = detected_arena_parameters['vertical_roads_under_construction']

		# print("Medicine Packages: ", medicine_package_details)
		# print("Traffic Signals: ", traffic_signals)
		# print("Start Node: ", start_node)
		# print("End Node: ", end_node)
		# print("Horizontal Roads under Construction: ", horizontal_roads_under_construction)
		# print("Vertical Roads under Construction: ", vertical_roads_under_construction)
		# print("\n\n")

	except Exception as e:
		print('Your task_1a.py throwed an Exception, kindly debug your code!\n')
		traceback.print_exc(file=sys.stdout)
		sys.exit()

	try:

		## Connect to CoppeliaSim arena
		coppelia_client = RemoteAPIClient()
		sim = coppelia_client.getObject('sim')

		## Define all models
		all_models = []

		## Setting up coppeliasim scene
		print("[1] Setting up the scene in CoppeliaSim")
		all_models = pb_theme.place_packages(medicine_package_details, sim, all_models)
		all_models = pb_theme.place_traffic_signals(traffic_signals, sim, all_models)
		all_models = pb_theme.place_horizontal_barricade(horizontal_roads_under_construction, sim, all_models)
		all_models = pb_theme.place_vertical_barricade(vertical_roads_under_construction, sim, all_models)
		all_models = pb_theme.place_start_end_nodes(start_node, end_node, sim, all_models)
		print("[2] Completed setting up the scene in CoppeliaSim")
		print("[3] Checking arena configuration in CoppeliaSim")

	except Exception as e:
		print('Your task_4a.py throwed an Exception, kindly debug your code!\n')
		traceback.print_exc(file=sys.stdout)
		sys.exit()

	pb_theme.send_message_via_socket(connection_1, "CHECK_ARENA")

	## Check if arena setup is ok or not
	message = pb_theme.receive_message_via_socket(connection_1)
	while True:


		if message == "ARENA_SETUP_OK":
			print("[4] Arena was properly setup in CoppeliaSim")
			break
		elif message == "ARENA_SETUP_NOT_OK":
			print("[4] Arena was not properly setup in CoppeliaSim")
			connection_1.close()
			# connection_2.close()
			server.close()
			sys.exit()
		else:
			pass

	## Send Start Simulation Command to PB_Socket
	pb_theme.send_message_via_socket(connection_1, "SIMULATION_START")

	## Check if simulation started correctly
	message = pb_theme.receive_message_via_socket(connection_1)
	while True:
		# message = pb_theme.receive_message_via_socket(connection_1)

		if message == "SIMULATION_STARTED_CORRECTLY":
			print("[5] Simulation was started in CoppeliaSim")
			break

		if message == "SIMULATION_NOT_STARTED_CORRECTLY":
			print("[5] Simulation was not started in CoppeliaSim")
			sys.exit()

	## Send Start Command to Raspberry Pi to start execution
	pb_theme.send_message_via_socket(connection_2, "START")


	task_4b_implementation(sim)

	## Send Stop Simulation Command to PB_Socket
	pb_theme.send_message_via_socket(connection_1, "SIMULATION_STOP")

	## Check if simulation started correctly
	message = pb_theme.receive_message_via_socket(connection_1)
	while True:
		# message = pb_theme.receive_message_via_socket(connection_1)

		if message == "SIMULATION_STOPPED_CORRECTLY":
			print("[6] Simulation was stopped in CoppeliaSim")
			break

		if message == "SIMULATION_NOT_STOPPED_CORRECTLY":
			print("[6] Simulation was not stopped in CoppeliaSim")
			sys.exit()
