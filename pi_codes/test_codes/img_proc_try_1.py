import numpy as np
import cv2

cap = cv2.VideoCapture(r'video2.mp4')

while(1):

    ret, frame = cap.read()
    cv2.imshow('Frame', frame)
    img = frame.copy()
    img2 = frame.copy()

    ### exposure correction ###
    contrast = 5. # Contrast control ( 0 to 127)
    brightness = 2. # Brightness control (0-100)
    img = cv2.addWeighted( img, contrast, img, 0, brightness)
    twoDimage = img.reshape((-1,3))
    twoDimage = np.float32(twoDimage)

    ### clustering ###
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    attempts=0
    ret,label,center=cv2.kmeans(twoDimage,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    #cv2.imshow('Output', result_image)

    ### centroid detection ###
    gray_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY_INV)
    M = cv2.moments(thresh)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 320, 350
    cv2.circle(img2, (cX, cY), 5, (255, 255, 255), -1)
    cv2.putText(img2, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    ### draw line and find theta ###
    mp = (320,480)
    mpup = (320,350)
    if cX == mp[0]:
        cX = cX+0.01
    m = (cY - mp[1])/(cX-mp[0])
    theta = round(np.arctan(1/m)*(180/3.14))
    cv2.circle(img2,mp, 5, (255, 255, 255), -1)
    cv2.putText(img2, str(theta), (mp[0] , mp[1] - 50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    img2 = cv2.line(img2, (int(cX), int(cY)), mp, (255, 255, 255), 2)
    img2 = cv2.line(img2, mpup, mp, (0, 255, 0), 2)

    cv2.imshow("Final",img2)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

