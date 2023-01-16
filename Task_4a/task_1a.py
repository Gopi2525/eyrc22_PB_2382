'''
*****************************************************************************************
*
*        		===============================================
*           		Pharma Bot (PB) Theme (eYRC 2022-23)
*        		===============================================
*
*  This script is to implement Task 1A of Pharma Bot (PB) Theme (eYRC 2022-23).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:			2382
# Author List:  Aakash,Devaprasad,Ilamthendral,Gopi
# Filename:			task_1a.py
# Functions:		detect_traffic_signals, detect_horizontal_roads_under_construction, detect_vertical_roads_under_construction,
#					detect_medicine_packages, detect_arena_parameters
# 					[ Comma separated list of functions in this file ]


####################### IMPORT MODULES #######################
## You are not allowed to make any changes in this section. ##
## You have to implement this task with the three available ##
## modules for this task (numpy, opencv)                    ##
##############################################################
import cv2
import numpy as np
##############################################################

################# ADD UTILITY FUNCTIONS HERE #################





##############################################################

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

if __name__ == "__main__":

    # path directory of images in test_images folder
	img_dir_path = "public_test_images/"

    # path to 'maze_0.png' image file
	file_num = 0
	img_file_path = img_dir_path + 'maze_' + str(file_num) + '.png'
	
	# read image using opencv
	maze_image = cv2.imread(img_file_path)
	
	print('\n============================================')
	print('\nFor maze_' + str(file_num) + '.png')

	# detect and print the arena parameters from the image
	arena_parameters = detect_arena_parameters(maze_image)

	print("Arena Prameters: " , arena_parameters)

	# display the maze image
	cv2.imshow("image", maze_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	choice = input('\nDo you want to run your script on all test images ? => "y" or "n": ')
	
	if choice == 'y':

		for file_num in range(1, 15):
			
			# path to maze image file
			img_file_path = img_dir_path + 'maze_' + str(file_num) + '.png'
			
			# read image using opencv
			maze_image = cv2.imread(img_file_path)
	
			print('\n============================================')
			print('\nFor maze_' + str(file_num) + '.png')
			
			# detect and print the arena parameters from the image
			arena_parameters = detect_arena_parameters(maze_image)

			print("Arena Parameter: ", arena_parameters)
				
			# display the test image
			cv2.imshow("image", maze_image)
			cv2.waitKey(2000)
			cv2.destroyAllWindows()