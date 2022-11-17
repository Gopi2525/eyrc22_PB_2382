import cv2
import numpy as np

def centroid_finder(image):

	img = image

	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17) 

	lower_bound = np.array([74, 74, 74])
	upper_bound = np.array([160,160, 160])
	image = cv2.inRange(image, lower_bound, upper_bound)


	blur = cv2.GaussianBlur(image,(13,13),0)
	thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)[1]

	image = thresh

	blurImg = cv2.blur(image,(10,10)) 

	kernel = np.array([[0, -1, 0],[-1, 5,-1],[0, -1, 0]])
	image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
	image = image_sharp

	contours, hierarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	contours = sorted(contours, key = cv2.contourArea, reverse = True) [:1]

	for c in contours:
	   # calculate moments for each contour
	   M = cv2.moments(c)
	 
	   # calculate x,y coordinate of center
	   cX = int(M["m10"] / M["m00"])
	   cY = int(M["m01"] / M["m00"])
	   cv2.circle(img, (cX, cY), 5, (0, 255, 0), -1)
	   cv2.putText(img, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	return(img)

cap = cv2.VideoCapture('output.mp4')
if (cap.isOpened()== False):
    print("Error opening video file")
 
while(cap.isOpened()):
     
    ret, frame = cap.read()
    if ret == True:

    	image = centroid_finder(frame)
    	cv2.imshow("Image", image)

    	if cv2.waitKey(75) & 0xFF == ord('q'):
    		break

    else:
    	break

cap.release()
 
cv2.destroyAllWindows()
