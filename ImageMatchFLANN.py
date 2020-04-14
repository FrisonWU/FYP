#
'''
基于FLANN的匹配器(FLANN based Matcher)
1.FLANN代表近似最近邻居的快速库。它代表一组经过优化的算法，用于大数据集中的快速最近邻搜索以及高维特征。
2.对于大型数据集，它的工作速度比BFMatcher快。
3.需要传递两个字典来指定要使用的算法及其相关参数等
对于SIFT或SURF等算法，可以用以下方法：
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
对于ORB，可以使用以下参数：
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12   这个参数是searchParam,指定了索引中的树应该递归遍历的次数。值越高精度越高
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
'''
import cv2 as cv
from matplotlib import pyplot as plt

queryImage=cv.imread("key template.jpg",0)
trainingImage=cv.imread("key target.jpg",0)#读取要匹配的灰度照片
sift=cv.xfeatures2d.SIFT_create()#创建sift检测器
kp1, des1 = sift.detectAndCompute(queryImage,None)
kp2, des2 = sift.detectAndCompute(trainingImage,None)

# 暴力匹配BFMatcher，遍历描述符，确定描述符是否匹配，然后计算匹配距离并排序

#BFMatcher函数参数：
# normType：NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2。
# NORM_L1和NORM_L2是SIFT和SURF描述符的优先选择，NORM_HAMMING和NORM_HAMMING2是用于ORB算
'''
bf = cv.BFMatcher(normType=cv.NORM_L2, crossCheck=True)
matches = bf.knnMatch(des1, des2, k=2)
'''

#设置Flannde参数

FLANN_INDEX_KDTREE=0
FLANN_INDEX_KDTREE = 1,
FLANN_INDEX_KMEANS = 2,
FLANN_INDEX_COMPOSITE = 3,
FLANN_INDEX_KDTREE_SINGLE = 4,
FLANN_INDEX_HIERARCHICAL = 5,
FLANN_INDEX_LSH = 6,
FLANN_INDEX_SAVED = 254,
FLANN_INDEX_AUTOTUNED = 255,

indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
searchParams= dict(checks=50)
flann=cv.FlannBasedMatcher(indexParams,searchParams)
matches=flann.knnMatch(des1,des2,k=2)


#设置好初始匹配值
matchesMask=[[0,0] for i in range (len(matches))]
for i, (m,n) in enumerate(matches):
	if m.distance< 0.54*n.distance: #舍弃小于0.5的匹配结果
		matchesMask[i]=[1,0]
drawParams=dict(matchColor=(0,0,255),singlePointColor=(255,0,0),matchesMask=matchesMask,flags=0) #给特征点和匹配的线定义颜色
resultimage=cv.drawMatchesKnn(queryImage,kp1,trainingImage,kp2,matches,None,**drawParams) #画出匹配的结果
plt.imshow(resultimage,),plt.show()
