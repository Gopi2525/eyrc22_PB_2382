'''
*****************************************************************************************
*
*        		===============================================
*           		Pharma Bot (PB) Theme (eYRC 2022-23)
*        		===============================================
*
*  This script is to implement Task 3C of Pharma Bot (PB) Theme (eYRC 2022-23).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:			[ Team-ID ]
# Author List:		[  Aakash K , Devaprasad S , Gopi M , Ilam Thendral R  ]
# Filename:			task_3c.py
# Functions:		[ perspective_transform, transform_values, set_values ]
# 					


####################### IMPORT MODULES #######################
## You are not allowed to make any changes in this section. ##
## You have to implement this task with the five available  ##
## modules for this task                                    ##
##############################################################
import cv2 
import numpy 
from  numpy import interp
from zmqRemoteApi import RemoteAPIClient
import zmq
##############################################################

#################################  ADD UTILITY FUNCTIONS HERE  #######################


def und_undwrm(img):

    h = 316
    w = 467
    cameraMatrix = numpy.array( [[432.64843938,0.,323.04192955],[0.,431.83992831,245.06752113],[0.,0.,1.]])
    newCameraMatrix = numpy.array([[239.79382324,0.,342.27052461],[0.,245.33255005,268.8847273 ],[0.,0.,1.]])
    roi = (104, 105, 467, 316)
    dist = numpy.array([[-0.40167891,0.19897568,0.00285976,0.00067859,-0.04539993]])

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
    `image` :	[ numpy array ]
            numpy array of image returned by cv2 library 

    Returns:
    ---
    `warped_image` : [ numpy array ]
            return cropped arena image as a numpy array
    
    Example call:
    ---
    warped_image = perspective_transform(image)
    """   
    warped_image = [] 

#################################  ADD YOUR CODE HERE  ###############################
    
    img_sent = image.copy()
    ArUco_details_dict, ArUco_corners = task_1b.detect_ArUco_details(img_sent)

    img  = image

    img_copy = numpy.copy(img)
    img_copy = cv2.cvtColor(img_copy,cv2.COLOR_BGR2RGB)

    for key in range(1,5):

        o = checkKey(ArUco_details_dict, key)

        if o == 1:
            continue

        else:
            
            print("Aruco number",key,"is missing [from perspective_transform function]")#end="\r"

    if (len(ArUco_details_dict) >= 4):

        k = ArUco_details_dict
        l = ArUco_corners

    else:

        y = -1

        ###### Doubt :( 
        while (len(Storage.data_orientation[y])<4):

            y =y-1


        k = Storage.data_orientation[y]
        l = Storage.data_corners[y]

    pt_A = [k[1][0][0], k[1][0][1]]
    pt_B = [k[2][0][0], k[2][0][1]]
    pt_C = [k[3][0][0], k[3][0][1]]
    pt_D = [k[4][0][0], k[4][0][1]]

    width_AD = numpy.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = numpy.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = numpy.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = numpy.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    input_pts = numpy.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = numpy.float32([[0, 0],[0, maxHeight - 1],[maxWidth - 1, maxHeight - 1],[maxWidth - 1, 0]])

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
    `image` :	[ numpy array ]
            numpy array of image returned by camera

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
    ArUco_details_dict, ArUco_corners = task_1b.detect_ArUco_details(image_sended)

    if (len(ArUco_details_dict) >= 4):

        Storage.data_orientation.append(k)
        Storage.data_corners.append(l)

    else :

        k = Storage.data_orientation[-1]
        l = Storage.data_corners[-1]


    x1y1 = k[1][0]
    x2y2 = k[2][0]
    x3y3 = k[3][0]
    x4y4 = k[4][0]

    if len(ArUco_details_dict) == 5:

        x5y5 = k[5][0]

        polygon_data = (x1y1, x2y2, x3y3,x4y4)
        centroid_point = centroid(polygon_data) 

        xcg = -centroid_point[0]
        ycg = -centroid_point[1]

        T = [[1, 0, 0, xcg],[0, 1, 1, ycg],[0, 0, 0, 0],[0, 0, 0, 1]]
        pb = [[x5y5[0]], [x5y5[1]], [0], [1]]

        T = numpy.array(T,ndmin=4)
        pb = numpy.array(pb,ndmin=4)

        result=numpy.matmul(T,pb) 

        xgr = result[0][0][0][0]
        ygr = result[0][0][1][0]


        """
        sx = int(240/(x4y4[0]-x3y3[0]))
        sy = int(240/(x4y4[1]-x3y3[1]))
        """

        sx = -30
        sy = -30       
        
        
        """
        sx = int(240/316)
        sy = int(240/417)
        """

        #print(image.shape)

        S = [[sx,0],[0,sy]]
        P = [[xgr],[ygr]]

        S = numpy.array(S)
        P = numpy.array(P)

        result=numpy.matmul(T,pb) 

        xgr = (result[0][0][0][0]/100)
        ygr = (result[0][0][1][0]/100)

        scene_parameters = [xgr,ygr,ArUco_details_dict[5][1]]

    else:

        scene_parameters = [2,2,0]

        print("x5y5 is missing")

        
######################################################################################

    return scene_parameters

def set_values(scene_parameters):
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
    aruco_handle = sim.getObject('/aruco_5')
#################################  ADD YOUR CODE HERE  ###############################

    position_robot = (-scene_parameters[0]*0.765,scene_parameters[1]*0.72,+0.030)
    eulerAngles_robot = [0.0, 3.141592502593994,-((scene_parameters[2]*(3.141592502593994/180))-3.141592502593994)]
    
    sim.setObjectPosition(aruco_handle,sim.handle_parent, position_robot)
    sim.setObjectOrientation(aruco_handle,sim.handle_parent,eulerAngles_robot)

######################################################################################

    return None



if __name__ == "__main__":
    client = RemoteAPIClient()
    sim = client.getObject('sim')
    task_1b = __import__('task_1b')
#################################  ADD YOUR CODE HERE  ################################


    #####initialising camera #####
    vid = cv2.VideoCapture(0)

    ##### initializing storage #####
    Storage.dictonary_past_data()
  
    while(True):
          
        ret, frame = vid.read()
        dst,dst1 = und_undwrm(frame)

        img = frame.copy() # normal output
        img1 = frame.copy() # normal output copy
        frame1 = dst.copy() # un disorted 

        ArUco_details_dict, ArUco_corners = task_1b.detect_ArUco_details(frame)

        for key in range(1,6):

            o = checkKey(ArUco_details_dict, key)

            if o == 1:
                
                t = "Storage.data_"+str(key)
                t.append(k[key])

            else:




                

        

        #img = task_1b.mark_ArUco_image(img, ArUco_details_dict, ArUco_corners)
        img = task_1b.mark_ArUco_image(img, k, l) 

        #warped_image_1 = perspective_transform(frame) # normal output as input : frame
        warped_image_2 = perspective_transform(frame1) # undisorted image as input : frame 1

        warped_image_2_copy = warped_image_2.copy()

        #ArUco_details_dict_5, ArUco_corners_5 = task_1b.detect_ArUco_details(warped_image_2_copy)
        #print(len(ArUco_details_dict_5))

        scene_parameters = transform_values(warped_image_2_copy)
        print(scene_parameters)

        cv2.imshow('frame',img)
        #cv2.imshow('frame_dist',frame1)

        #cv2.imshow('warped_image_of_output',warped_image_1)
        cv2.imshow('warped_image_of_undisorted',warped_image_2)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):

            break

      
    vid.release()
    cv2.destroyAllWindows()


#######################################################################################
