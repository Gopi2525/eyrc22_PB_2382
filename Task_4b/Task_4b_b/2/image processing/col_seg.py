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

def yellow_region(img):

    image_in = img.copy()
    image_in = cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB); # Load the glared image
    h, s, v = cv2.split(cv2.cvtColor(image_in, cv2.COLOR_RGB2HSV)) # split into HSV components

    nonSat = s > 50 # Find all pixels that are not very saturated

    # Slightly decrease the area of the non-satuared pixels by a erosion operation.
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    nonSat = cv2.erode(nonSat.astype(np.uint8), disk)
    #nonSat = nonSat.astype(np.uint8)

    # Set all brightness values, where the pixels are still saturated to 0.
    v2 = v.copy()
    v2[nonSat == 0] = 0

    blurred = cv2.GaussianBlur(v2, (11, 11), 0)
    thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)[1]

    #result = cv2.bitwise_and(img,img, mask= thresh)

    return thresh

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

        yellow_reg = yellow_region(img1)

        # Display the resulting frame
        cv2.imshow('Frame',frame)
        cv2.imshow('thresh',yellow_reg)


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