#opencv----特征匹配----BFMatching
import cv2
import numpy as np
from matplotlib import pyplot as plt
#读取需要特征匹配的两张照片，格式为灰度图。
img2 = cv2.imread('key template3.jpg',cv2.IMREAD_GRAYSCALE)
f2 = img2>np.mean(img2)
f22 = img2<=np.mean(img2)
img2[f2]=255
img2[f22]=0
img2 = cv2.resize(img2,dsize=(int(img2.shape[1]/2),int(img2.shape[0]/2)))

img1 = img2.copy()
cv2.imshow("1",img1)
#img1 = cv2.resize(img1,dsize=(int(img1.shape[1]/2),int(img1.shape[0]/2)))
img2 = cv2.imread('key target 6.jpg',cv2.IMREAD_GRAYSCALE)
f2 = img2>np.mean(img2)
f22 = img2<=np.mean(img2)
img2[f2]=255
img2[f22]=0
img2 = cv2.resize(img2,dsize=(int(img2.shape[1]/2),int(img2.shape[0]/2)))
cv2.imshow("2",img2)

orb=cv2.ORB_create()#建立orb特征检测器
kp1,des1=orb.detectAndCompute(img1,None)#计算template中的特征点和描述符
kp2,des2=orb.detectAndCompute(img2,None) #计算target中的
bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True) #建立匹配关系
mathces=bf.match(des1,des2) #匹配描述符
mathces=sorted(mathces,key=lambda x:x.distance) #据距离来排序
result= cv2.drawMatches(img1,kp1,img2,kp2,mathces[:11],None,flags=2) #画出匹配关系
plt.imshow(result),plt.show() #matplotlib描绘出来
