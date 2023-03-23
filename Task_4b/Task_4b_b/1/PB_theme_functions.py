'''
*****************************************************************************************
*
*        		     ===============================================
*           		       Pharma Bot (PB) Theme (eYRC 2022-23)
*        		     ===============================================
*
*  This script contains all the past implemented functions of Pharma Bot (PB) Theme
*  (eYRC 2022-23).
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:			[ PB 2382 ]
# Author List:		[ Aakash K , Devaprasad S , Gopi M , Ilam Thendral R]
# Filename:			PB_theme_functions.py
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
from cv2 import aruco
from pyzbar.pyzbar import decode
import json
##############################################################

################# ADD UTILITY FUNCTIONS HERE #################

def centroid(vertexes):
     x_list = vertexes[0][0] + vertexes[0][2] 
     y_list = vertexes[1][0] + vertexes[1][2]
     length = len(vertexes)
     """
     x = sum(x_list) / length
     y = sum(y_list) / length
     """
     x = x_list / length
     y = y_list / length
     return(int(x), int(y))

def detect_ArUco(img):

    Detected_ArUco_markers = {}
    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
    parameters = aruco.DetectorParameters_create()
    corners,ids, _ = aruco.detectMarkers(img,aruco_dict,parameters=parameters)

    for i in range(0,ids.shape[0]):
        Detected_ArUco_markers[ids[i][0,]] = corners[i][0,]

    return Detected_ArUco_markers

def Calculate_orientation_in_degree(Detected_ArUco_markers,img):

    ArUco_marker_angles = {}
    cnt = 0
    ans_x = 0
    ans_y = 0
    midpoint_x = 0
    midpoint_y = 0
    right_x = 0
    right_y = 0
    for i in Detected_ArUco_markers:
        for j in Detected_ArUco_markers[i]:
            if (cnt%4) == 0:
                ans_x = j[0,]
                ans_y = j[1,]
                midpoint_x = ans_x
                midpoint_y = ans_y
                cnt = 1
            else:
                cnt += 1
                if cnt == 2:
                    midpoint_x = (midpoint_x + j[0,])/2
                    midpoint_y = (midpoint_y + j[1,])/2
                    right_x = j[0,]
                    right_y = j[1,]
                ans_x += j[0,]
                ans_y += j[1,]
        ans_x = int(ans_x/4)
        ans_y = int(ans_y/4)
        midpoint_x = int(midpoint_x)
        midpoint_y = int(midpoint_y)
        cv2.circle(img,(ans_x,ans_y), 5, (0,0,255), -1)
        cv2.line(img,(ans_x,ans_y),(midpoint_x,midpoint_y),(255,0,0),5)
        midpoint_x = midpoint_x - ans_x
        midpoint_y = -(midpoint_y - ans_y)

        ans_x = 0
        ans_y = 0
        id_str = str(i)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if midpoint_y < 0:
            li = int((360-np.arccos(np.inner([1,0],[midpoint_x,midpoint_y])/np.linalg.norm([midpoint_x,midpoint_y]))*180/np.pi))
            li = li - 90
            
            if li <= 180 :
                li = li
            else:
                li = li - 360
            
            ang = str(li),
            ArUco_marker_angles[i] = ang
        else:
            le = int((np.arccos(np.inner([1,0],[midpoint_x,midpoint_y])/np.linalg.norm([midpoint_x,midpoint_y]))*180/np.pi))
            le = le - 90
            
            if le <= 180 :
                le = le
            else:
                le = -(le-360)
            
            ang = str(le),
            ArUco_marker_angles[i] = ang
    return ArUco_marker_angles

##############################################################

def detect_Qr_details(image):

    """
    Purpose:
    ---
    This function takes the image as an argument and returns a dictionary such
    that the message encrypted in the Qr code is the key and the center
    co-ordinates of the Qr code is the value, for each item in the dictionary

    Input Arguments:
    ---
    `image` :   [ numpy array ]
            numpy array of image returned by cv2 library
    Returns:
    ---
    `Qr_codes_details` : { dictionary }
            dictionary containing the details regarding the Qr code
    
    Example call:
    ---
    Qr_codes_details = detect_Qr_details(image)
    """    
    Qr_codes_details = {}

    ##############  ADD YOUR CODE HERE  ##############

    center_pts = []
    data_in_qr = []

    data = decode(image)
    print(data)

    for i in range(0,len(data)):

        (p,q,r,s) = data[i].polygon
        #print(p,q,r,s)
        x_l = [p[0],q[0],r[0],s[0]]
        y_l = [p[1],q[1],r[1],s[1]]

        center = centroid([x_l,y_l])
        center_pts.append(center)

        t = data[i].data
        print(t.decode())
        data_in_qr.append(t.decode())

        Qr_codes_details[data_in_qr[i]] = list(center_pts[i])
        
    ##################################################
    
    return Qr_codes_details    

def detect_ArUco_details(image):

    """
    Purpose:
    ---
    This function takes the image as an argument and returns a dictionary such
    that the id of the ArUco marker is the key and a list of details of the marker
    is the value for each item in the dictionary. The list of details include the following
    parameters as the items in the given order
        [center co-ordinates, angle from the vertical, list of corner co-ordinates] 
    This order should be strictly maintained in the output

    Input Arguments:
    ---
    `image` :   [ numpy array ]
            numpy array of image returned by cv2 library
    Returns:
    ---
    `ArUco_details_dict` : { dictionary }
            dictionary containing the details regarding the ArUco marker
    
    Example call:
    ---
    ArUco_details_dict = detect_ArUco_details(image)
    """    
    ArUco_details_dict = {} #should be sorted in ascending order of ids
    ArUco_corners = {}
    
    ##############  ADD YOUR CODE HERE  ##############

    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
    parameters = aruco.DetectorParameters_create()
    corners,ids, _ = aruco.detectMarkers(image,aruco_dict,parameters=parameters)

    corner_points = []
    center_points = []


    if len(corners) > 0:

        ids = ids.flatten()

        for (markerCorner, markerID) in zip(corners, ids):

            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)

            corner_points.append([topLeft, topRight, bottomRight, bottomLeft])
            center_points.append([cX,cY])
            
            #cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)

        ids = ids.tolist()
        Ar_markers = detect_ArUco(image)
        Orientations = Calculate_orientation_in_degree(Ar_markers,image)

        for i in range(0,len(ids)):

            a = Orientations[ids[i]]
            ArUco_details_dict[ids[i]] = [center_points[i],int(a[0])] 

        ArUco_corners = Ar_markers
    
    ##################################################
    
    return ArUco_details_dict, ArUco_corners 

######### YOU ARE NOT ALLOWED TO MAKE CHANGES TO THE CODE BELOW #########   

# marking the Qr code with center and message

def mark_Qr_image(image, Qr_codes_details):
    for message, center in Qr_codes_details.items():
        encrypted_message = message
        x_center = int(center[0])
        y_center = int(center[1])
        
        cv2.circle(img, (x_center, y_center), 5, (0,0,255), -1)
        cv2.putText(image,str(encrypted_message),(x_center + 20, y_center+ 20),cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)

    return image

# marking the ArUco marker with the center, angle and corners

def mark_ArUco_image(image,ArUco_details_dict, ArUco_corners):

    for ids, details in ArUco_details_dict.items():
        center = details[0]
        cv2.circle(image, center, 5, (0,0,255), -1)

        corner = ArUco_corners[int(ids)]
        cv2.circle(image, (int(corner[0][0]), int(corner[0][1])), 5, (50, 50, 50), -1)
        cv2.circle(image, (int(corner[1][0]), int(corner[1][1])), 5, (0, 255, 0), -1)
        cv2.circle(image, (int(corner[2][0]), int(corner[2][1])), 5, (128, 0, 255), -1)
        cv2.circle(image, (int(corner[3][0]), int(corner[3][1])), 5, (255, 255, 255), -1)

        tl_tr_center_x = int((corner[0][0] + corner[1][0]) / 2)
        tl_tr_center_y = int((corner[0][1] + corner[1][1]) / 2) 

        cv2.line(image,center,(tl_tr_center_x, tl_tr_center_y),(255,0,0),5)
        display_offset = 2*int(math.sqrt((tl_tr_center_x - center[0])**2+(tl_tr_center_y - center[1])**2))
        cv2.putText(image,str(ids),(center[0]+int(display_offset/2),center[1]),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        angle = details[1]
        cv2.putText(image,str(angle),(center[0]-display_offset,center[1]),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    return image


def shapes_rgb(a):
    
    shapes = []
    for i in range(0,len(a)):
        shapes.append(a[i][0].partition('_')[-1])
    return shapes

def colours_rgb(a):

    colour = []
    for i in range(0,len(a)):
        colour.append(a[i][0][:a[i][0].index("_")])
    return colour

def detect_Qr_details_drop(image):

    Qr_codes_details = []
    data = decode(image)

    if len(data) != 0 :
        t = data[0].data
        qr_message = t.decode()
    else :
        qr_message = None

    result = qr_message.split('"')[1::2]
    for i in range(0,len(result),2):

        a = [result[i],result[i+1]]
        Qr_codes_details.append(a) 

    return Qr_codes_details

def read_package_details(medicine_package_details):

    package_name = []
    
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
            model = colour+three_d_shape(shape)
            package_name.append(str(shop)+"_"+colour+three_d_shape(shape))

    return package_name

def centroid(vertexes):
     x_list = vertexes[0][0] + vertexes[0][2] 
     y_list = vertexes[1][0] + vertexes[1][2]
     length = len(vertexes)

     x = x_list / length
     y = y_list / length
     return(int(x), int(y))

def detect_Qr_details(image):

    Qr_codes_details = {}

    center_pts = []
    data_in_qr = []

    data = decode(image)

    for i in range(0,len(data)):

        (p,q,r,s) = data[i].polygon
        #print(p,q,r,s)
        x_l = [p[0],q[0],r[0],s[0]]
        y_l = [p[1],q[1],r[1],s[1]]

        center = centroid([x_l,y_l])
        center_pts.append(center)

        t = data[i].data
        #print(t.decode())
        data_in_qr.append(t.decode())

        Qr_codes_details[data_in_qr[i]] = list(center_pts[i])
            
    return Qr_codes_details

def position_data(node):
    
    d = {"A1":[-0.89,-0.9],"B1":[-0.51,-0.9],"C1":[-0.2,-0.9],"D1":[0.15,-0.9],"E1":[0.5,-0.9],"F1":[0.8,-0.9],"A2":[-0.89,-0.51],"B2":[-0.57,-0.51],"C2":[-0.2,-0.51],"D2":[0.15,-0.51],"E2":[0.5,-0.51],"F2":[0.8,-0.51],"A3":[-0.89,-0.12],"B3":[-0.57,-0.12],"C3":[-0.2,-0.12],"D3":[0.15,-0.12],"E3":[0.5,-0.12],"F3":[0.8,-0.12],"A4":[-0.89,0.22],"B4":[-0.57,0.22],"C4":[-0.2,0.22],"D4":[0.15,0.22],"E4":[0.57,0.22],"F4":[0.8,0.22],"A5":[-0.89,0.57],"B5":[-0.57,0.57],"C5":[-0.2,0.57],"D5":[0.15,0.57],"E5":[0.5,0.57],"F5":[0.8,0.57],"A6":[-0.89,0.9],"B6":[-0.57,0.9],"C6":[-0.2,0.9],"D6":[0.15,0.9],"E6":[0.5,0.9],"F6":[0.8,0.9]}
    
    return d[node]

##############################################################


################## ADD SOCKET COMMUNICATION ##################
####################### FUNCTIONS HERE #######################
"""
Add functions written in Task 3D for setting up a Socket
Communication Server in this section
"""

def setup_server(host, port):

	"""
	Purpose:
	---
	This function creates a new socket server and then binds it
	to a host and port specified by user.

	Input Arguments:
	---
	`host` :	[ string ]
			host name or ip address for the server

	`port` : [ string ]
			integer value specifying port name
	Returns:

	`server` : [ socket object ]
	---


	Example call:
	---
	server = setupServer(host, port)
	"""

	server = None

	##################	ADD YOUR CODE HERE	##################

	s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
	print('Socket created')

	s.bind((host,port))
	print('Socket bind complete')

	server = s

	##########################################################

	return server

def setup_connection(server):
	"""
	Purpose:
	---
	This function listens for an incoming socket client and
	accepts the connection request

	Input Arguments:
	---
	`server` :	[ socket object ]
			socket object created by setupServer() function
	Returns:
	---
	`server` : [ socket object ]

	Example call:
	---
	connection = setupConnection(server)
	"""
	connection = None
	address = None

	##################	ADD YOUR CODE HERE	##################

	server.listen(5)
	print('Socket now listening')

	connection,address = server.accept()

	##########################################################

	return connection, address

def receive_message_via_socket(connection):
	"""
	Purpose:
	---
	This function listens for a message from the specified
	socket connection and returns the message when received.

	Input Arguments:
	---
	`connection` :	[ connection object ]
			connection object created by setupConnection() function
	Returns:
	---
	`message` : [ string ]
			message received through socket communication

	Example call:
	---
	message = receive_message_via_socket(connection)
	"""

	message = None

	##################	ADD YOUR CODE HERE	##################

	message = connection.recv(1024)
	message = message.decode("utf-8")

	##########################################################

	return message

def send_message_via_socket(connection, message):
	"""
	Purpose:
	---
	This function sends a message over the specified socket connection

	Input Arguments:
	---
	`connection` :	[ connection object ]
			connection object created by setupConnection() function

	`message` : [ string ]
			message sent through socket communication

	Returns:
	---
	None

	Example call:
	---
	send_message_via_socket(connection, message)
	"""

	##################	ADD YOUR CODE HERE	##################

	connection.send(bytes(message,"utf-8"))

	##########################################################

##############################################################
##############################################################

######################### ADD TASK 2B ########################
####################### FUNCTIONS HERE #######################
"""
Add functions written in Task 2B for reading QR code from
CoppeliaSim arena in this section
"""

def read_qr_code(sim):
	"""
	Purpose:
	---
	This function detects the QR code present in the CoppeliaSim vision sensor's
	field of view and returns the message encoded into it.

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

##############################################################
##############################################################

############### ADD ARENA PARAMETER DETECTION ################
####################### FUNCTIONS HERE #######################
"""
Add functions written in Task 1A and 3A for detecting arena parameters
from configuration image in this section
"""

def detect_all_nodes(image):

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
     vertical_roads_under_construction.append(one+'-'+two)
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
     horizontal_roads_under_construction.append(one+'-'+two)
   x=x+100;y=y+100
  x1=x1+100;y1=y1+100
  
 return horizontal_roads_under_construction


def detect_medicine_packages_present(maze_image):   

 medicine_packages = []
 def shopsorter(shop):

  none=[]
  a = len(shop)
  maper = []
  b =[]
  c = []
  sortedshop =[]
  if a == 0 :
      return none 
  else :
    for i in range(0,a):
        if i%3 == 0 :
         p = shop[i]
         po = ord(p[0])
         maper.append(po)
         maper.append(i)
         b.append(po)

    b.sort()

    for i in range(0,len(b)):
         t = b[i]
         s = maper.index(t)
         al = s+1
         c.append(maper[al])

    for i in range(0,len(c)):
        sortedshop = sortedshop + [shop[c[i]],shop[c[i]+1],shop[c[i]+2]]

    return sortedshop

    
 col=[];color=[];l=[];n=0;x1=150;y1=193;x2=107;y2=150;x3=150;y3=193;x4=107;y4=150
 part=[];x=107;y=193;final=[];s=[];sac=[];pre=[]
 image = maze_image
 for i  in range(1,7):
  crop=image[107:150,x4:y4]
  lower_bound = np.array([0, 255, 0])
  upper_bound = np.array([0,255,0])
  green = cv2.inRange(crop, lower_bound, upper_bound)
  if np.count_nonzero(green) !=0 :
   col.append('Green')
  lower_bound = np.array([255, 255, 0])
  upper_bound = np.array([255,255,0])
  blue = cv2.inRange(crop, lower_bound, upper_bound)
  if np.count_nonzero(blue) !=0 :
   col.append('Skyblue')
  lower_bound = np.array([180, 0, 255])
  upper_bound = np.array([180,0,255])
  pink= cv2.inRange(crop, lower_bound, upper_bound)
  if np.count_nonzero(pink) !=0 :
   col.append('Pink')
  lower_bound = np.array([0, 127, 255])
  upper_bound = np.array([0,127,255])
  orange= cv2.inRange(crop, lower_bound, upper_bound)
  if np.count_nonzero(orange) !=0 :
   col.append('Orange')
  x4=x4+100;y4=y4+100

  crop=image[107:150,x3:y3]
  lower_bound = np.array([0, 255, 0])
  upper_bound = np.array([0,255,0])
  green = cv2.inRange(crop, lower_bound, upper_bound)
  if np.count_nonzero(green) !=0 :
   col.append('Green')
  lower_bound = np.array([255, 255, 0])
  upper_bound = np.array([255,255,0])
  blue = cv2.inRange(crop, lower_bound, upper_bound)
  if np.count_nonzero(blue) !=0 :
   col.append('Skyblue')
  lower_bound = np.array([180, 0, 255])
  upper_bound = np.array([180,0,255])
  pink= cv2.inRange(crop, lower_bound, upper_bound)
  if np.count_nonzero(pink) !=0 :
   col.append('Pink')
  lower_bound = np.array([0, 127, 255])
  upper_bound = np.array([0,127,255])
  orange= cv2.inRange(crop, lower_bound, upper_bound)
  if np.count_nonzero(orange) !=0 :
   col.append('Orange')
  x3=x3+100;y3=y3+100
 
  crop=image[150:193,x2:y2]
  lower_bound = np.array([0, 255, 0])
  upper_bound = np.array([0,255,0])
  green = cv2.inRange(crop, lower_bound, upper_bound)
  if np.count_nonzero(green) !=0 :
   col.append('Green')
  lower_bound = np.array([255, 255, 0])
  upper_bound = np.array([255,255,0])
  blue = cv2.inRange(crop, lower_bound, upper_bound)
  if np.count_nonzero(blue) !=0 :
    col.append('Skyblue')
  lower_bound = np.array([180, 0, 255])
  upper_bound = np.array([180,0,255])
  pink= cv2.inRange(crop, lower_bound, upper_bound)
  if np.count_nonzero(pink) !=0 :
   col.append('Pink')
  lower_bound = np.array([0, 127, 255])
  upper_bound = np.array([0,127,255])
  orange= cv2.inRange(crop, lower_bound, upper_bound)
  if np.count_nonzero(orange) !=0 :
    col.append('Orange')
  x2=x2+100;y2=y2+100
 
  crop=image[150:193,x1:y1]
  lower_bound = np.array([0, 255, 0])
  upper_bound = np.array([0,255,0])
  green = cv2.inRange(crop, lower_bound, upper_bound)
  if np.count_nonzero(green) !=0 :
   col.append('Green')
  lower_bound = np.array([255, 255, 0])
  upper_bound = np.array([255,255,0])
  blue = cv2.inRange(crop, lower_bound, upper_bound)
  if np.count_nonzero(blue) !=0 :
   col.append('Skyblue')
  lower_bound = np.array([180, 0, 255])
  upper_bound = np.array([180,0,255])
  pink= cv2.inRange(crop, lower_bound, upper_bound)
  if np.count_nonzero(pink) !=0 :
   col.append('Pink')
  lower_bound = np.array([0, 127, 255])
  upper_bound = np.array([0,127,255])
  orange= cv2.inRange(crop, lower_bound, upper_bound)
  if np.count_nonzero(orange) !=0 :
   col.append('Orange')
  x1=x1+100;y1=y1+100
 

 gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 image = maze_image
 gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 _, thrash = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY)
 contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
 for i,contour in enumerate(contours):
  shape = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
  cnt = contours[i]
  M = cv2.moments(cnt)
  area = cv2.contourArea(cnt)
  if i!=0:
   if 600>area>300:
    if M['m00'] != 0:
      cx = int(M['m10']/M['m00'])
      cy = int(M['m01']/M['m00'])
    if len(shape)==12:
       final.append('Circle')
       mo1=cx,cy
       s.append(mo1)
    if len(shape)==8:
       final.append('Square')
       mo2=cx,cy
       s.append(mo2)
    if len(shape)==5:
       final.append('Triangle')
       mo3=cx,cy-1
       s.append(mo3)
  x=x+100;y=y+100

 final1=list(reversed(final))
 s=list(map(lambda el:[el], s))

 

 for i in range(len(s)):  
    for j in s[i]:
        if type(j) is tuple:
            s[i] = list(j)

 shapey=list(reversed(s))
 x1=107;y1=193;no=[]
 for i  in range(1,7):
  crop=image[107:193,x1:y1]
  gray_image = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
  _, thrash = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY)
  contours, hierarchy = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  no.append(len(contours)-1)
  x1=x1+100;y1=y1+100

 shop0=[];shop1=[];shop2=[];shop3=[];shop4=[];shop5=[];ls=0;i=0
 for j in range(0,len(no)):
   for i in range(i,no[j]+i):
    if j==0:
     shop0.append(col[i])
     shop0.append(final1[i])
     shop0.append(shapey[i])
    if j==1:
     shop1.append(col[i])
     shop1.append(final1[i])
     shop1.append(shapey[i])
    if j==2:
     shop2.append(col[i])
     shop2.append(final1[i])
     shop2.append(shapey[i])
    if j==3:
     shop3.append(col[i])
     shop3.append(final1[i])
     shop3.append(shapey[i])
    if j==4:
     shop4.append(col[i])
     shop4.append(final1[i])
     shop4.append(shapey[i])
    if j==5:
     shop5.append(col[i])
     shop5.append(final1[i])
     shop5.append(shapey[i])
   ls=ls+no[j]
   i=ls
 

 shop0=shopsorter(shop0)
 shop1=shopsorter(shop1)
 shop2=shopsorter(shop2)
 shop3=shopsorter(shop3)
 shop4=shopsorter(shop4)
 shop5=shopsorter(shop5)


 if  len(shop0)!=0:
  for i in range(0,len(shop0)):
   if i %4==0:
    shop0.insert(i,'Shop_1')


 if  len(shop1)!=0:
  for i in range(0,len(shop1)):
   if i %4==0:
    shop1.insert(i,'Shop_2')

 if  len(shop2)!=0:
  for i in range(0,len(shop2)):
   if i %4==0:
    shop2.insert(i,'Shop_3')


 if  len(shop3)!=0:
  for i in range(0,len(shop3)):
   if i %4==0:
    shop3.insert(i,'Shop_4')

 if  len(shop4)!=0:
  for i in range(0,len(shop4)):
   if i %4==0:
    shop4.insert(i,'Shop_5')

 if  len(shop5)!=0:
  for i in range(0,len(shop5)):
   if i %4==0:
    shop5.insert(i,'Shop_6')

 medicine_packages_present=[]
 medicine_packages_present.append(shop0)
 medicine_packages_present.append(shop1)
 medicine_packages_present.append(shop2)
 medicine_packages_present.append(shop3)
 medicine_packages_present.append(shop4)
 
	
 return medicine_packages_present

def detect_arena_parameters(maze_image):

 arena_parameters = {}
 start_node,end_node,traffic_signals= detect_all_nodes(maze_image)
 horizontal_roads_under_construction = detect_horizontal_roads_under_construction(maze_image)
 vertical_roads_under_construction = detect_vertical_roads_under_construction(maze_image)
 medicine_packages_present = detect_medicine_packages_present(maze_image)
 Arena_parameters={'start_node':start_node,'end_node':end_node,'traffic_signals':traffic_signals,'horizontal_roads_under_construction':horizontal_roads_under_construction,'vertical_roads_under_construction':vertical_roads_under_construction,'medicine_packages':medicine_packages_present}
	
 return Arena_parameters
##############################################################
##############################################################

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


####################### ADD ARENA SETUP ######################
####################### FUNCTIONS HERE #######################
"""
Add functions written in Task 4A for setting up the CoppeliaSim
Arena according to the configuration image in this section
"""

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

        package_name = []

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
            package_name.append(colour+three_d_shape(shape))

        for l in range(0,len(paths)):

            position_of_package  = (pos_of_package(shop_reference,l)[0],pos_of_package(shop_reference,l)[1],0.02)
            objectHandle_med_package = sim.loadModel(paths[l])
            sim.setObjectAlias(objectHandle_med_package,package_name[l])
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

        position_of_traffic_signal = (coordinates_of_signal(i)[0],coordinates_of_signal(i)[1],0.15588)
        objectHandle_traffic_signal = sim.loadModel(traffic_sig_model)
        sim.setObjectAlias(objectHandle_traffic_signal,"Signal_"+i)
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

    position_of_start_signal = (coordinates_of_signal(start_node)[0],coordinates_of_signal(start_node)[1],0.15588)
    position_of_end_signal = (coordinates_of_signal(end_node)[0],coordinates_of_signal(end_node)[1],0.15588)

    objectHandle_start = sim.loadModel(start_node_model)
    objectHandle_end = sim.loadModel(end_node_model)

    sim.setObjectAlias(objectHandle_start,"Start_Node")
    sim.setObjectAlias(objectHandle_end,"End_Node")

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
        sim.setObjectAlias(objectHandle_horizontal_baricade,"Horizontal_missing_road_"+i[0:2]+"_"+i[3:])
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
        sim.setObjectAlias(objectHandle_vertical_baricade,"Vertical_missing_road_"+i[0:2]+"_"+i[3:])
        sim.setObjectParent(objectHandle_vertical_baricade,arena,1)
        sim.setObjectPosition(objectHandle_vertical_baricade,sim.handle_parent, position_of_vertical_baricade)
        all_models.append(objectHandle_vertical_baricade)

        n = n+1

	####################################################################

    return all_models

##############################################################
##############################################################
