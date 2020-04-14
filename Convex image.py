import cv2
import numpy as np
import matplotlib.pyplot as plt
    #from skimage import data,color,morphology,feature

#生成二值测试图像
'''1、加载图片&同态滤波'''
img= cv2.imread('star_template.jpg',cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img,dsize=(int(img1.shape[1]/2),int(img1.shape[0]/2)))
edgs= feature.canny(img, sigma=3, low_threshold=10, high_threshold=50)

chull = morphology.convex_hull_object(edgs)

#绘制轮廓
fig, axes = plt.subplots(1,2,figsize=(8,8))
ax0, ax1= axes.ravel()
ax0.imshow(edgs,plt.cm.gray)
ax0.set_title('many objects')
ax1.imshow(chull,plt.cm.gray)
ax1.set_title('convex_hull image')
plt.show()