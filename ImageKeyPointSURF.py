
'''
SURF算法找特征点
'''
import cv2
import numpy as np

img1 = cv2.imread('key template.jpg')
img2 = cv2.imread('key target.jpg')
#img = cv2.resize(img,dsize=(600,400))
#转换为灰度图像
#gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
#gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
#创建一个SURF对象
surf = cv2.xfeatures2d.SURF_create(1800)
#SIFT对象会使用Hessian算法检测关键点，并且对每个关键点周围的区域计算特征向量。该函数返回关键点的信息和描述符
keypoints1,descriptor1 = surf.detectAndCompute(img1,None)
keypoints2,descriptor2 = surf.detectAndCompute(img2,None)
#在图像上绘制关键点
img1 = cv2.drawKeypoints(image=img1,keypoints = keypoints1,outImage=img1,color=(255,0,0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2 = cv2.drawKeypoints(image=img2,keypoints = keypoints2,outImage=img2,color=(255,0,0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#显示图像
cv2.imshow('template_keypoints',img1)
cv2.imshow('target_keypoints',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()