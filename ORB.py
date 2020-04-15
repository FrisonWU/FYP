import cv2
import numpy as np

def Boundbox(src, template):  # find the mass center of the target in picture
    binary = src.copy()
    cv2.bitwise_not(binary, binary)
    num_labels, label, area_status, area_center = cv2.connectedComponentsWithStats(binary, connectivity=8,
                                                                                   ltype=cv2.CV_32S)  # calculate the connected component of picture
    # cv2.imshow('bin',binary)
    # avg_min = 20
    for item in range(num_labels):
        LEFT = area_status[item, cv2.CC_STAT_LEFT]
        TOP = area_status[item, cv2.CC_STAT_TOP]
        WIDTH = area_status[item, cv2.CC_STAT_WIDTH]
        HEIGHT = area_status[item, cv2.CC_STAT_HEIGHT]
        #img_bin = cv2.rectangle(binary, (LEFT, TOP), (LEFT + WIDTH, TOP + HEIGHT), (0, 255, 0), 3)
        img_cropped = binary[TOP:TOP + HEIGHT, LEFT:LEFT + WIDTH]
        blank = np.zeros((HEIGHT+100,WIDTH+100),dtype=img_cropped.dtype)
        blank[50:50+HEIGHT,50:50+WIDTH]=img_cropped
        img_cropped = blank

        if WIDTH < 5 or HEIGHT < 5:
            continue
        if WIDTH < 100 or HEIGHT < 100:
            img_cropped = cv2.resize(img_cropped, dsize=(int(img_cropped.shape[1] * 7), int(img_cropped.shape[0] * 7)))
        orb = cv2.ORB_create()  # 建立orb特征检测器
        kp1, des1 = orb.detectAndCompute(template, None)  # 计算template中的特征点和描述符
        kp2, des2 = orb.detectAndCompute(img_cropped, None)  # 计算target中的

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # 建立匹配关系

        if des2 is None:
            continue
        mathces = bf.match(des1, des2)  # 匹配描述符
        mathces = sorted(mathces, key=lambda x: x.distance)  # 据距离来排序
        avg_min = 20
        # item_min = 0
        '''寻找ORB特征匹配对应的distance平均值最小值'''
        dis = 0
        avg = 0
        match_num = len(mathces)
        if not area_status[item, cv2.CC_STAT_AREA] == np.max(area_status[:, 4]):
            if match_num >= 30:
                for i in range(30):
                    dis += mathces[i].distance
                avg = dis / match_num
            if avg < avg_min:
                item_min = item
                avg_min = avg
        result = cv2.drawMatches(template, kp1, img_cropped, kp2, mathces[:11], None, flags=2)  # 画出匹配关系
    cv2.imshow('ORB',result)
    print(avg_min)
    bincopy = np.array(binary)
    bincopy[label == item_min] = 255
    bincopy[label != item_min] = 0
    return bincopy

def Homography(img1,img2):
    orb =  cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)  # 对两个字符串进行异或运算，并统计结果为1的个数，那么这个数就是汉明距离
    try:
        # Match descriptors.
        matches = bf.match(des1, des2)
        # 由于匹配顺序是：matches = bf.match(des1,des2)，先des1后des2。
        # 因此，kp1的索引由DMatch对象属性为queryIdx决定，kp2的索引由DMatch对象属性为trainIdx决定
        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)  # 按distance排序

        img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:80], None, flags=2)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
        # print(src_pts)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.5)
        # print(mask.shape)
        # matchesMask = mask.ravel().tolist()

        # self.observation = np.array([[H[0, 0]],[H[0, 1]],[H[0, 2]],[H[1, 0]],
        #  [H[1, 1]],[H[1, 2]]])
        # print(self.observation)

        print(H)
        cv2.imshow('homo',img3)
    except:
        print("Less than 10 matches")

def MassCenter(src, OutputImage, size):
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
if __name__ == '__main__':
    i=1
    cap = cv2.VideoCapture(0)
    ret,frame = cap.read()
    temp = cv2.imread('temp_invbin.jpg',cv2.IMREAD_GRAYSCALE)
    #cv2.imshow('temp',temp)
    temp = cv2.resize(temp,dsize=(int(temp.shape[1]/2),int(temp.shape[0]/2)))
    # tar = cv2.imread('tar.jpg',cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('tar',tar)
    # ret, tar = cv2.threshold(tar, 200, 255, cv2.THRESH_BINARY)  # set threshold and obtain binary image
    # Boundbox(tar,temp)
    preframe = frame
    while True:
        #preframe = frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret,bin = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
        #cv2.imshow('bincam',bin)
        imgbin=Boundbox(bin,temp)
        cv2.imshow('bin',imgbin)
        if i % 10 == 0:
            preframe = frame
        i = i+1
        cv2.imshow('preframe',preframe)
        Homography(preframe,frame)
        print(i)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()