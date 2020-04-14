import cv2
import numpy as np

import matplotlib.pyplot as plt
#plt.switch_backend('agg')

def Boundbox (src,template,max_area=0,max_label=0,left=0,top=0,width=0,height=0): #find the mass center of the target in picture
    gray = src.copy()
    ret, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    cv2.bitwise_not(binary, binary)
    #cv2.bitwise_not(template,template)
    num_labels,label,area_status,area_center=cv2.connectedComponentsWithStats(binary, connectivity=8, ltype=cv2.CV_32S) #calculate the connected component of picture
    item_min = 0
    avg_min = 20
    for item in range(num_labels):
        LEFT = area_status[item, cv2.CC_STAT_LEFT]
        TOP = area_status[item, cv2.CC_STAT_TOP]
        WIDTH = area_status[item, cv2.CC_STAT_WIDTH]
        HEIGHT = area_status[item, cv2.CC_STAT_HEIGHT]
        #img_bin = cv2.rectangle(binary, (LEFT, TOP), (LEFT + WIDTH, TOP + HEIGHT), (0, 255, 0), 3)
        #img_cropped = img_bin.crop((LEFT,TOP,LEFT + WIDTH,TOP + HEIGHT))
        img_cropped = binary[TOP:TOP+HEIGHT,LEFT:LEFT+WIDTH]
        cv2.imshow('Original',binary)
        #cv2.imshow('imbin',img_bin)
        cv2.imshow('Cropped', img_cropped)

        if WIDTH < 5 or HEIGHT < 5:
            continue
        if WIDTH < 100 or HEIGHT <100:
            img_cropped = cv2.resize(img_cropped, dsize=(int(img_cropped.shape[1] * 7), int(img_cropped.shape[0] * 7)))
            cv2.imshow('Cropped_Update', img_cropped)
        orb = cv2.ORB_create()  # 建立orb特征检测器
        kp1, des1 = orb.detectAndCompute(template, None)  # Compute feature points of star template
        kp2, des2 = orb.detectAndCompute(img_cropped, None)  # Compute feature points in each label

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # BF matcher

        if des2 is None:
            continue
        mathces = bf.match(des1, des2)  # BF MATCHING OF two descripters
        mathces = sorted(mathces, key=lambda x: x.distance)  # arrage feature point based on BF distance
        result = cv2.drawMatches(template, kp1, img_cropped, kp2, mathces[:11], None, flags=2)  # Draw matching line
        # plt.imshow(result), plt.show()  # matplotlib描绘出来
        print('Processing item' + str(item))
        #avg_min = 100
        #item_min = 0
        '''寻找ORB特征匹配对应的distance平均值最小值'''
        dis = 0
        avg = 150
        match_num = len(mathces)
        sample_num = 40
        if not area_status[item, cv2.CC_STAT_AREA] == np.max(area_status[:, 4]):
            if match_num >= sample_num:
                for i in range(sample_num):
                    dis += mathces[i].distance
                avg = dis / sample_num
                if avg < avg_min:
                    item_min = item
                    avg_min = avg

        cv2.imshow("ORB", result)
    print(item_min)
    print(avg_min)
        #cv2.destroyAllWindows()
    bincopy = np.array(binary)
    bincopy[label == item_min] = 255
    bincopy[label != item_min] = 0
    return bincopy

def ConnectComponent(src, i=0, max_area=0, max_label=0, left=0, top=0, width=0,height=0):  # find the
    gray = src.copy()
    ret, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    num_labels, label, area_status, area_center = cv2.connectedComponentsWithStats(binary, connectivity=8,ltype=cv2.CV_32S)  # calculate the connected component of picture

    max = np.max(area_status, axis=0)[4]
    for item in range(num_labels):
        label_area = area_status[i, cv2.CC_STAT_AREA]

        if max_area < label_area:
            if i != 0:
                if label_area != max:  # not the background value
                    max_area = label_area
                    max_label = i
        i += 1
    max_label = 9
    LEFT = area_status[max_label, cv2.CC_STAT_LEFT]
    TOP = area_status[max_label, cv2.CC_STAT_TOP]
    WIDTH = area_status[max_label, cv2.CC_STAT_WIDTH]
    HEIGHT = area_status[max_label, cv2.CC_STAT_HEIGHT]
    bincopy = np.array(binary)
    bincopy[label == max_label] = 255
    bincopy[label != max_label] = 0
    img_bin = cv2.rectangle(bincopy, (LEFT, TOP), (LEFT + WIDTH, TOP + HEIGHT), (0, 255, 0), 3)
    img_cropped = img_bin[TOP:TOP + HEIGHT, LEFT:LEFT + WIDTH]
    return bincopy,img_cropped

def ConnectComponentTarget(src, i=0, max_area=0, max_label=0, left=0, top=0, width=0,height=0):  # find the
    gray = src.copy()
    ret, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    #cv2.imshow('threshold',binary)
    num_labels, label, area_status, area_center = cv2.connectedComponentsWithStats(binary, connectivity=8,ltype=cv2.CV_32S)  # calculate the connected component of picture

    max = np.max(area_status, axis=0)[4]
    for item in range(num_labels):
        label_area = area_status[i, cv2.CC_STAT_AREA]

        if max_area < label_area:
            if i != 0:
                if label_area != max:  # not the background value
                    max_area = label_area
                    max_label = i
        i += 1
    max_label = 2
    LEFT = area_status[max_label, cv2.CC_STAT_LEFT]
    TOP = area_status[max_label, cv2.CC_STAT_TOP]
    WIDTH = area_status[max_label, cv2.CC_STAT_WIDTH]
    HEIGHT = area_status[max_label, cv2.CC_STAT_HEIGHT]
    bincopy = np.array(binary)
    bincopy[label == max_label] = 255
    bincopy[label != max_label] = 0
    img_bin = cv2.rectangle(bincopy, (LEFT, TOP), (LEFT + WIDTH, TOP + HEIGHT), (0, 255, 0), 3)
    img_cropped = img_bin[TOP:TOP + HEIGHT, LEFT:LEFT + WIDTH]
    return bincopy,img_cropped

def MassCenter(src, OutputImage,size):
    image_matrix = np.mat(src)
    m00 = image_matrix.sum()
    wx = [i for i in range(0, src.shape[0])]
    wy = [i for i in range(0, src.shape[1])]
    m10_xx = np.dot(wx, image_matrix)
    m01_yy = np.dot(image_matrix, wy)
    # m10_x = np.average(image_matrix, axis=0, weights=wx)
    # m01_y = np.average(image_matrix, axis=1, weights=wy)
    m01 = m01_yy.sum()
    m10 = m10_xx.sum()
    xcenter = int(round(m10 / m00))
    ycenter = int(round(m01 / m00))
    x_start = int(xcenter - (size / 2))
    x_end = int(xcenter + (size / 2))
    y_start = int(ycenter - (size / 2))
    y_end = int(ycenter + (size / 2))
    src = cv2.line(OutputImage, (y_start, xcenter), (y_end, xcenter), (255, 0, 0), 3)
    src = cv2.line(OutputImage, (ycenter, x_start), (ycenter, x_end), (255, 0, 0), 3)
    cv2.imshow('Mass center', OutputImage)

'''1、加载图片&同态滤波'''

img1= cv2.imread('star_template.jpg',cv2.IMREAD_GRAYSCALE)
img1 = cv2.resize(img1,dsize=(int(img1.shape[1]/2),int(img1.shape[0]/2)))
img2 = cv2.imread('star1.jpg',cv2.IMREAD_GRAYSCALE)
img2 = cv2.resize(img2,dsize=(int(img2.shape[1]/2),int(img2.shape[0]/2)))
img3= cv2.imread('star2.jpg',cv2.IMREAD_GRAYSCALE)
img3 = cv2.resize(img3,dsize=(int(img3.shape[1]/2),int(img3.shape[0]/2)))
img4 = cv2.imread('star3.jpg',cv2.IMREAD_GRAYSCALE)
img4 = cv2.resize(img4,dsize=(int(img4.shape[1]/2),int(img4.shape[0]/2)))
img4 = cv2.imread('temp.jpg',cv2.IMREAD_GRAYSCALE)
img4 = cv2.resize(img4,dsize=(int(img4.shape[1]/2),int(img4.shape[0]/2)))
cv2.imshow('original',img2)
ret, img2_bin = cv2.threshold(img2, 120, 255, cv2.THRESH_BINARY)
cv2.imshow('thres',img2_bin)
src = img3
'''2、Connect Component'''
binpro,binprochop = ConnectComponent(img2)
cv2.imshow('bincopy',binpro)
cv2.bitwise_not(binpro,binpro)
cv2.imshow('binive',binpro)
bininv,binchop = ConnectComponentTarget(binpro)
cv2.imshow('bininvpro',bininv)
cv2.imshow('binchop',binchop)
binchop = cv2.resize(binchop, dsize=(int(binchop.shape[1] * 7), int(binchop.shape[0] * 7)))
cv2.imwrite('template.jpg',binchop)

'''3、Improved Connected Components'''
impro=Boundbox(src,binchop)
cv2.imshow('impro',impro)

''''4、Mass Centre'''
MassCenter(impro,src,20)
cv2.waitKey(0)
cv2.destroyAllWindows()
