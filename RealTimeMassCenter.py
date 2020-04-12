import cv2
import numpy as np
from Image_track import Kalman
import matplotlib.pyplot as plt
class RealtimeMassCenter:
    def __init__(self):
        self.avg_min = 100
        self.avg = 150
        self.flag = False
        self.xcenter = [320]
        self.ycenter = [240]
        self.item_min = 0

    def Boundbox (self,src,template): #find the mass center of the target in picture
        avg_min = 100
        item_min =0
        binary = src.copy()
        #gray = src.copy()
        #ret, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
        cv2.bitwise_not(binary, binary)
        num_labels,label,area_status,area_center=cv2.connectedComponentsWithStats(binary, connectivity=8, ltype=cv2.CV_32S) #calculate the connected component of picture
        for item in range(num_labels):
            LEFT = area_status[item, cv2.CC_STAT_LEFT]
            TOP = area_status[item, cv2.CC_STAT_TOP]
            WIDTH = area_status[item, cv2.CC_STAT_WIDTH]
            HEIGHT = area_status[item, cv2.CC_STAT_HEIGHT]
            img_bin = cv2.rectangle(binary, (LEFT, TOP), (LEFT + WIDTH, TOP + HEIGHT), (0, 255, 0), 3)
            #img_cropped = img_bin.crop((LEFT,TOP,LEFT + WIDTH,TOP + HEIGHT))
            img_cropped = img_bin[TOP:TOP+HEIGHT,LEFT:LEFT+WIDTH]

            if WIDTH < 5 or HEIGHT < 5:
                continue
            if WIDTH < 100 or HEIGHT <100:
                img_cropped = cv2.resize(img_cropped, dsize=(int(img_cropped.shape[1] * 7), int(img_cropped.shape[0] * 7)))
            orb = cv2.ORB_create()  # 建立orb特征检测器
            kp1, des1 = orb.detectAndCompute(template, None)  # 计算template中的特征点和描述符
            kp2, des2 = orb.detectAndCompute(img_cropped, None)  # 计算target中的

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # 建立匹配关系

            if des2 is None:
                continue
            mathces = bf.match(des1, des2)  # 匹配描述符
            mathces = sorted(mathces, key=lambda x: x.distance)  # 据距离来排序
            #result = cv2.drawMatches(template, kp1, img_cropped, kp2, mathces[:11], None, flags=2)  # 画出匹配关系
            item_min = 0
            '''寻找ORB特征匹配对应的distance平均值最小值'''
            dis = 0
            #avg_min = 100
            match_num = len(mathces)
            sample_num = 25
            if not area_status[item,cv2.CC_STAT_AREA] == np.max(area_status[:,4]):
                if match_num>=sample_num:
                    for i in range(sample_num):
                        dis += mathces[i].distance
                    self.avg = dis / sample_num
                    if self.avg < avg_min:
                        self.flag = False
                        item_min = item
                        avg_min = self.avg
        if avg_min >= self.avg_min-0.05:
            self.flag = True
        self.avg_min = avg_min
        print(self.avg_min)
        #print(item_min)
        bincopy = np.array(binary)
        bincopy[label == item_min] = 255
        bincopy[label != item_min] = 0
        return bincopy

    def MassCenter(self,src):
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
        mx = m10/m00
        my = m01/m00
        if mx is not None:
            xcenter = int(round(mx))
        if my is not None:
            ycenter = int(round(my))
        if self.flag == False and xcenter >=0 and ycenter>=0:
            self.xcenter.append(xcenter)
            self.ycenter.append(ycenter)
        elif self.flag == True:
            print('Xcenter is %d',self.xcenter[-1])
            print('Ycenter is %d',self.ycenter[-1])
            self.xcenter.append(self.xcenter[-1])
            self.ycenter.append(self.ycenter[-1])
        xcenter = int(self.xcenter[-1])
        ycenter = int(self.ycenter[-1])
        #print('X original: %d',xcenter)
        return src,xcenter,ycenter
        #cv2.imshow('Mass center', OutputImage)

    def Mass_center_Draw(self,OutputImage, xcenter,ycenter,size):
        x_start = int(xcenter - (size / 2))
        x_end = int(xcenter + (size / 2))
        y_start = int(ycenter - (size / 2))
        y_end = int(ycenter + (size / 2))
        OutputImage = cv2.line(OutputImage, (y_start, xcenter), (y_end, xcenter), (255, 0, 0), 3)
        OutputImage = cv2.line(OutputImage, (ycenter, x_start), (ycenter, x_end), (255, 0, 0), 3)
        return OutputImage

    def Mass_cennter_Track(self,OutputImage,obx,oby,flx,fly):
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

if __name__ == '__main__':
    template=cv2.imread('template.jpg')
    cap = cv2.VideoCapture(0)
    # 从视频流循环帧
    lk = Kalman()
    Rt = RealtimeMassCenter()
    dt = 1/30
    time = 0
    timeList,originalx,filteredx,originaly,filteredy = [],[],[],[],[]
    while True:
        time += dt
        timeList.append(time)
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #ret, binary = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY)
        '''Connected Component'''
        cam = Rt.Boundbox(gray,template)
        cv2.imshow("Frame", cam)
        fps = cap.get(cv2.CAP_PROP_FPS)  # 帧速率
        ''''3、Mass Centre'''
        mass,xc,yc=Rt.MassCenter(cam)
        originalx.append(xc)
        originaly.append(yc)
        xk,yk=lk.start(xc,yc)
        filteredx.append(xk)
        filteredy.append(yk)

        mc=Rt.Mass_center_Draw(gray,int(xk),int(yk),20)
        cv2.imshow("Window", mc)
        # 退出：Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            break
    print('FPS is %d',fps)
    plt.subplot(2,1,1)
    plt.title('XResult Analysis')
    plt.plot(timeList, originalx, color='green', label='Observed X')
    plt.plot(timeList, filteredx, color='red', label='Kalman Filtered X')
    plt.legend()  # 显示图例
    plt.xlabel('iteration times')
    plt.ylabel('value')
    plt.subplot(2,1,2)
    plt.title('yResult Analysis')
    plt.plot(timeList, originaly, color='blue', label='Observed Y')
    plt.plot(timeList, filteredy, color='skyblue', label='Kalman Filtered Y')
    plt.legend()  # 显示图例
    plt.xlabel('iteration times')
    plt.ylabel('value')
    plt.show()
    # 清理窗口
    cv2.destroyAllWindows()