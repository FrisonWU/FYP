# -*- coding: utf-8 -*-
import cv2

import numpy as np
#import pdb
#pdb.set_trace()#turn on the pdb prompt

img = cv2.imread('key target.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)

cv2.drawKeypoints(gray,kp,img)

cv2.imwrite('sift_keypoints.jpg',img)
cv2.imshow("sift_keypoint",img)
cv2.waitKey(0)
cv2.destroyAllWindows()