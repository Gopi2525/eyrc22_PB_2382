import cv2
import numpy as np

def corners(img,a):

    if a == 0 :
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    else:
        gray = img

    gray = np.float32(gray)
    corners = cv2.goodFeaturesToTrack(gray, 200, 0.1, 25)
    corners = np.int0(corners)


    for corner in corners:
        x,y = corner.ravel()
        cv2.circle(img,(x,y),3,255,-1)

    return img



# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('road.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        img1 = frame.copy()
        img2 = frame.copy()

        #corner_img = corners(img1,0)

        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)[1]
        dst = cv2.inpaint(img2,thresh,3,cv2.INPAINT_NS)       

        # Display the resulting frame
        cv2.imshow('Frame',frame)
        #cv2.imshow('Blurred',blurred)
        #cv2.imshow('Thresh',thresh)
        cv2.imshow('dst',dst)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else: 
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()