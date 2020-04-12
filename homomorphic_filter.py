import cv2
import numpy as np
import matplotlib.pyplot as plt
def homomorphic_filter(src, d0=10, r1=0.5, rh=2, c=4, h=2.0, l=0.5):

    gray = src.copy()
    #if len(src.shape) > 2:
    #    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    #gray = np.float64(gray)
    rows, cols = gray.shape

    gray_fft = np.fft.fft2(gray)
    gray_fftshift = np.fft.fftshift(gray_fft)
    dst_fftshift = np.zeros_like(gray_fftshift)
    M, N = np.meshgrid(np.arange(-cols // 2, cols // 2), np.arange(-rows // 2, rows // 2))
    D = np.sqrt(M ** 2 + N ** 2)

    Z = (rh - r1) * (1 - np.exp(-c * (D ** 2 / d0 ** 2))) + r1
    dst_fftshift = Z * gray_fftshift
    dst_fftshift = (h - l) * dst_fftshift + l
    dst_ifftshift = np.fft.ifftshift(dst_fftshift)
    dst_ifft = np.fft.ifft2(dst_ifftshift)
    dst = np.real(dst_ifft)
    dst = np.uint8(np.clip(dst, 0, 255,out=dst))

    plt.figure(1)
    plt.title("Fitler Relation")
    plt.plot(D,Z)
    return dst



'''1、加载图片&同态滤波'''
img1= cv2.imread('star_template.jpg',cv2.IMREAD_GRAYSCALE)
img1 = cv2.resize(img1,dsize=(int(img1.shape[1]/2),int(img1.shape[0]/2)))
img2 = cv2.imread('star1.jpg',cv2.IMREAD_GRAYSCALE)
img2 = cv2.resize(img2,dsize=(int(img2.shape[1]/2),int(img2.shape[0]/2)))
cv2.imwrite('out.jpg',img1)
img_filtered1 = homomorphic_filter(img1).copy()
img_filtered2 = homomorphic_filter(img2).copy()
cv2.imshow('gray1',img1)
plt.figure(2)
plt.subplot(1,2,1)
plt.imshow(img1)
plt.subplot(1,2,2)
plt.imshow(img_filtered1)



'''2、提取特征点'''
#创建一个SURF对象
surf = cv2.xfeatures2d.SURF_create(8000)
#SIFT对象会使用Hessian算法检测关键点，并且对每个关键点周围的区域计算特征向量。该函数返回关键点的信息和描述符
keypoints1,descriptor1 = surf.detectAndCompute(img_filtered1,None)
keypoints2,descriptor2 = surf.detectAndCompute(img_filtered2,None)
#在图像上绘制关键点
img_filtered1 = cv2.drawKeypoints(image=img_filtered1,keypoints = keypoints1,outImage=img_filtered1,color=(255,0,255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_filtered2 = cv2.drawKeypoints(image=img_filtered2,keypoints = keypoints2,outImage=img_filtered2,color=(255,0,255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#显示图像

cv2.imshow ('surf_feature1',img_filtered1)
cv2.imshow ('surf_feature2',img_filtered2)

'''3、特征点匹配'''
matcher = cv2.FlannBasedMatcher()
matchePoints = matcher.match(descriptor1,descriptor2)
#print(type(matchePoints),len(matchePoints),matchePoints[0])

#提取强匹配特征点
minMatch = 1
maxMatch = 0
for i in range(len(matchePoints)):
    if  minMatch > matchePoints[i].distance:
        minMatch = matchePoints[i].distance
    if  maxMatch < matchePoints[i].distance:
        maxMatch = matchePoints[i].distance
print('最佳匹配值是:',minMatch)
print('最差匹配值是:',maxMatch)

#获取排雷在前边的几个最优匹配结果
goodMatchePoints = []
for i in range(len(matchePoints)):
    if  True or matchePoints[i].distance < minMatch + (maxMatch-minMatch)/2:
        goodMatchePoints.append(matchePoints[i])

#绘制最优匹配点
outImg = None
outImg = cv2.drawMatches(img1,keypoints1,img2,keypoints2,goodMatchePoints,outImg,matchColor=(0,255,0),flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
cv2.imshow('matche',outImg)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

