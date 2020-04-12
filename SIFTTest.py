import cv2
import numpy as np
img = cv2.imread('key template1.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imshow('imgshowing',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
image1 = cv2.drawKeypoints(image=imge,keypoints = keypoints1,outImage=image1,color=(255,0,255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#img=cv2.drawKeypoints(gray,kp,img)
cv2.imshow('sift_keypoints1',image1)
cv2.waitKey(0)
cv2.destroyAllWindows()