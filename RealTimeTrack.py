import cv2
import numpy as np
import matplotlib.pyplot as plt

class Project:
    def Boundbox(src, template):  # find the mass center of the target in picture
        binary = src.copy()
        cv2.bitwise_not(binary, binary)
        num_labels, label, area_status, area_center = cv2.connectedComponentsWithStats(binary, connectivity=8,ltype=cv2.CV_32S)  # calculate the connected component of picture
        #avg_min = 20
        for item in range(num_labels):
            LEFT = area_status[item, cv2.CC_STAT_LEFT]
            TOP = area_status[item, cv2.CC_STAT_TOP]
            WIDTH = area_status[item, cv2.CC_STAT_WIDTH]
            HEIGHT = area_status[item, cv2.CC_STAT_HEIGHT]
            img_bin = cv2.rectangle(binary, (LEFT, TOP), (LEFT + WIDTH, TOP + HEIGHT), (0, 255, 0), 3)
            img_cropped = img_bin[TOP:TOP + HEIGHT, LEFT:LEFT + WIDTH]

            if WIDTH < 5 or HEIGHT < 5:
                continue
            if WIDTH < 100 or HEIGHT < 100:
                img_cropped = cv2.resize(img_cropped,dsize=(int(img_cropped.shape[1] * 7), int(img_cropped.shape[0] * 7)))
            orb = cv2.ORB_create()  # 建立orb特征检测器
            kp1, des1 = orb.detectAndCompute(template, None)  # 计算template中的特征点和描述符
            kp2, des2 = orb.detectAndCompute(img_cropped, None)  # 计算target中的

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # 建立匹配关系

            if des2 is None:
                continue
            mathces = bf.match(des1, des2)  # 匹配描述符
            mathces = sorted(mathces, key=lambda x: x.distance)  # 据距离来排序
            # result = cv2.drawMatches(template, kp1, img_cropped, kp2, mathces[:11], None, flags=2)  # 画出匹配关系
            avg_min = 20
            #item_min = 0
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
        print(avg_min)
        bincopy = np.array(binary)
        bincopy[label == item_min] = 255
        bincopy[label != item_min] = 0
        return bincopy

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
        return src

    def KALMANMassCenter(src):
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
        return xcenter,ycenter

    def Mass_center_Draw(OutputImage, xcenter, ycenter,kalmanx,kalmany, size):
        x_start = int(xcenter - (size / 2))
        x_end = int(xcenter + (size / 2))
        y_start = int(ycenter - (size / 2))
        y_end = int(ycenter + (size / 2))

        kalmanx_start =  int(kalmanx - (size / 2))
        kalmanx_end = int(kalmanx + (size / 2))
        kalmany_start = int(kalmany - (size / 2))
        kalmany_end = int(kalmany + (size / 2))
        OutputImage = cv2.line(OutputImage, (y_start, xcenter), (y_end, xcenter), (255, 0, 0), 3)
        OutputImage = cv2.line(OutputImage, (ycenter, x_start), (ycenter, x_end), (255, 0, 0), 3)

        OutputImage = cv2.line(OutputImage, (kalmany_start, kalmanx), (kalmany_end, kalmanx), (0, 0, 255), 3)
        OutputImage = cv2.line(OutputImage, (kalmany, kalmanx_start), (kalmany, kalmanx_end), (0, 0, 255), 3)
        return OutputImage

    def Mass_cennter_Track(OutputImage,obx,oby,flx,fly):
        size = 10
        cnt = len(obx)
        obx = np.array(obx)
        oby = np.array(oby)
        flx = np.array(flx)
        fly = np.array(fly)
        for i in range (cnt):
            obx_start = int(obx[i] - (size / 2))
            obx_end = int(obx[i] + (size / 2))
            oby_start = int(oby[i] - (size / 2))
            oby_end = int(oby[i] + (size / 2))
            OutputImage = cv2.line(OutputImage, (oby_start, obx[i]), (oby_end, obx[i]), (255, 0, 0), 3)
            OutputImage = cv2.line(OutputImage, (oby[i], obx_start), (oby[i], obx_end), (255, 0, 0), 3)

            flx_start = int(flx[i] - (size / 2))
            flx_end = int(flx[i] + (size / 2))
            fly_start = int(fly[i] - (size / 2))
            fly_end = int(fly[i] + (size / 2))
            OutputImage = cv2.line(OutputImage, (fly_start, flx[i]), (fly_end, flx[i]), (0, 0, 255), 3)
            OutputImage = cv2.line(OutputImage, (fly[i], flx_end), (fly[i], flx_start), (0, 0, 255), 3)
            size+=1
        return OutputImage


