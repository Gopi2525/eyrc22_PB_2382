import cv2
import numpy as np
import time
import math

def blackdetector(img):

	lower_bound = np.array([140, 140, 140])
	upper_bound = np.array([160,160, 160])
	black = cv2.inRange(img, lower_bound, upper_bound)

	return black

def yellowdetector(img):

	lower_bound = np.array([0, 190, 230])
	upper_bound = np.array([30,220, 260])
	yellow = cv2.inRange(img, lower_bound, upper_bound)

	return yellow

def givelines(img):

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	(thresh, blackAndWhiteImage) = cv2.threshold(grayimage, 127, 255, cv2.THRESH_BINARY)
	edges = cv2.Canny(blackAndWhiteImage, 50, 150, apertureSize=3)
	lines = cv2.HoughLines(edges, 1, np.pi/180, 200)


	if isinstance(lines, np.ndarray) :

		for r_theta in lines:

			arr = np.array(r_theta[0], dtype=np.float64)
			r, theta = arr

			a = np.cos(theta)
			b = np.sin(theta)

			x0 = a*r
			y0 = b*r

			x1 = int(x0 + 1000*(-b))
			y1 = int(y0 + 1000*(a))

			x2 = int(x0 - 1000*(-b))
			y2 = int(y0 - 1000*(a))


			if theta >= 1.48353 and theta <= 1.65806 : # 85 to 95

				cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

			if (theta >= 0.698132 and theta <= 1.48353) or (theta >= 1.65806 and theta <= 2.26893 ) : # [40 to 85] and [95 to 130]

				cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

			else :

				cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)



		return img,lines

	else :

		lines = np.ndarray([0,0])

		return img,lines

cap = cv2.VideoCapture('output.mp4')
if (cap.isOpened()== False):
    print("Error opening video file")

s = []
sb = []
 
while(cap.isOpened()):
     
    ret, frame = cap.read()
    if ret == True:

    	cv2.imshow('Frame', frame)
    	y = yellowdetector(frame)
    	bk = blackdetector(frame)

    	#cv2.imshow('Y', y)
    	#cv2.imshow('Bk', bk)
    	n_yellow_pix = np.sum(y == 255)
    	n_black_pix = np.sum(bk == 255)
    	#print('Number of white pixels:', n_yellow_pix)
    	#print('Number of white pixels:', n_black_pix)
    	s.append(n_yellow_pix)
    	sb.append(n_black_pix)



    	if cv2.waitKey(75) & 0xFF == ord('q'):
    		break

    else:
    	break

print(s)
cap.release()
 
cv2.destroyAllWindows()
