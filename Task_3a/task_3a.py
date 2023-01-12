'''
*****************************************************************************************
*
*        		===============================================
*           		Pharma Bot (PB) Theme (eYRC 2022-23)
*        		===============================================
*
*  This script is to implement Task 3A of Pharma Bot (PB) Theme (eYRC 2022-23).
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
# Filename:			task_3a.py
# Functions:		detect_all_nodes, detect_horizontal_roads_under_construction, detect_vertical_roads_under_construction,
#					detect_paths_to_graph, detect_arena_parameters, path_planning_dj_algo
# 					[ Comma separated list of functions in this file ]

####################### IMPORT MODULES #######################
## You are not allowed to make any changes in this section. ##
## You have to implement this task with the three available ##
## modules for this task (numpy, opencv)                    ##
##############################################################
import numpy as np
import cv2
##############################################################

################# ADD UTILITY FUNCTIONS HERE #################

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
	`maze_image` :	[ numpy array ]
			numpy array of image returned by cv2 library
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
	`maze_image` :	[ numpy array ]
			numpy array of image returned by cv2 library
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
	`maze_image` :	[ numpy array ]
			numpy array of image returned by cv2 library
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

######### YOU ARE NOT ALLOWED TO MAKE CHANGES TO THIS FUNCTION #########	

if __name__ == "__main__":

	# # path directory of images
	img_dir_path = "test_images/"

	for file_num in range(0,10):
			
			img_key = 'maze_00' + str(file_num)
			img_file_path = img_dir_path + img_key  + '.png'
			# read image using opencv
			image = cv2.imread(img_file_path)
			
			# detect the arena parameters from the image
			arena_parameters = detect_arena_parameters(image)

			print('\n============================================')
			print("IMAGE: ", file_num)
			print(arena_parameters["start_node"], "->>> ", arena_parameters["end_node"] )


			# path planning and getting the moves
			back_path=path_planning(arena_parameters["paths"], arena_parameters["start_node"], arena_parameters["end_node"])
			moves=paths_to_moves(back_path,arena_parameters["traffic_signals"])

			print("PATH PLANNED: ", back_path)
			print("MOVES TO TAKE: ", moves)
			

			# display the test image
			cv2.imshow("image", image)
			cv2.waitKey(0)
			cv2.destroyAllWindows()