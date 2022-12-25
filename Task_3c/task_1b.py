'''
*****************************************************************************************
*
*        		===============================================
*           		Pharma Bot (PB) Theme (eYRC 2022-23)
*        		===============================================
*
*  This script is to implement Task 1B of Pharma Bot (PB) Theme (eYRC 2022-23).
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
# Filename:			task_1b.py
# Functions:		detect_Qr_details, detect_ArUco_details
# 					[ Comma separated list of functions in this file ]


####################### IMPORT MODULES #######################
## You are not allowed to make any changes in this section. ##
## You have to implement this task with the five available  ##
## modules for this task                                    ##
##############################################################
import numpy as np
import cv2
from cv2 import aruco
import math
from pyzbar import pyzbar
##############################################################

################# ADD UTILITY FUNCTIONS HERE #################

from pyzbar.pyzbar import decode
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
    `image` :	[ numpy array ]
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

    ##############	ADD YOUR CODE HERE	##############

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
    `image` :	[ numpy array ]
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
    
    ##############	ADD YOUR CODE HERE	##############

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

if __name__ == "__main__":

    # path directory of images in test_images folder
    img_dir_path = "public_test_cases/"

    # choose whether to test Qr or ArUco images
    choice = input('\nWhich images do you want to test ? => "q" or "a": ')

    if choice == 'q':

        marker = 'qr'

    else:

        marker = 'aruco'

    for file_num in range(0,2):
        img_file_path = img_dir_path +  marker + '_' + str(file_num) + '.png'

        # read image using opencv
        img = cv2.imread(img_file_path)

        print('\n============================================')
        print('\nFor '+ marker  +  str(file_num) + '.png')

        # testing for Qr images
        if choice == 'q':
            Qr_codes_details = detect_Qr_details(img)
            print("Detected details of Qr: " , Qr_codes_details)

            # displaying the marked image
            img = mark_Qr_image(img, Qr_codes_details)
            cv2.imshow("img",img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # testing for ArUco images
        else:    
            ArUco_details_dict, ArUco_corners = detect_ArUco_details(img)
            print("Detected details of ArUco: " , ArUco_details_dict)

            #displaying the marked image
            img = mark_ArUco_image(img, ArUco_details_dict, ArUco_corners)  
            cv2.imshow("img",img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
