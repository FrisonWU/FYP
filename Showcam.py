from PyQt5 import QtCore,QtGui,QtWidgets
from PyQt5.QtGui import QPen, QPainter, QBrush, QBitmap, QPixmap, QImage
from PyQt5.QtWidgets import QWidget, QMessageBox, QLabel
from PyQt5.QtCore import QPoint, Qt, QRect
import sys
import cv2
class Template:
    init = 0
    choice =1
    ok = 2

class Video (QLabel):
    def setTimer(self, onoff):
        if onoff:
            print('Open')
            flag = self.cap.open(self.CAM_NUM + cv2.CAP_DSHOW)  # Open CAM, cv2.CAP_DSHOW can restart camera without error
            #self.button_open_camera.setText('Disable Camera')k
            self.timer_camera.start(30)
        else:
            print('No')
            self.timer_camera.stop()
            self.cap.release()  # 释放视频流
            self.clear()  # 清空视频显示区域
            #self.button_open_camera.setText('Enable Camera')
        self.onoff = onoff

    def __init__(self,parent=None):
        self.pitch = 0
        self.yaw = 0
        super().__init__(parent) #父类的构造函数

        self.timer_camera = QtCore.QTimer()  # 定义定时器，用于控制显示视频的帧率
        self.cap = cv2.VideoCapture()  # 视频流
        self.CAM_NUM = 0  # 为0时表示视频流来自笔记本内置摄像头

        self.sub_slot()
        self.onoff = False
        self.pix = QPixmap()  # 实例化一个 QPixmap 对象
        self.startX = 0  # 截屏初始X坐标
        self.startY = 0  # 截屏初始Y坐标
        self.lastX = 0
        self.lastY = 0
        self.endX = 0  # 截屏结束X坐标
        self.endY = 0  # 截屏结束Y坐标
        self.isDrawing = False
        self.camImage = None

    def sub_slot(self):
        self.timer_camera.timeout.connect(self.show_camera)  # 若定时器结束，则调用show_camera()

    def show_camera(self):
        flag, self.image = self.cap.read()  # 从视频流中读
        show = cv2.resize(self.image, (640, 480))  # 把读到的帧的大小重新设置为 640x480

        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        self.camImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.repaint()
        #self.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage

    def mousePressEvent(self, event):
        if self.isDrawing and event.button() == Qt.LeftButton:
            self.mapToParent(QPoint(0, 0))
            self.startX = min(max(event.x() - 0, 0), self.geometry().right())
            self.startY = min(max(event.y() - 0, 0), self.geometry().bottom())
            print('startX is %d', self.startX)
            print('startY is %d', self.startY)

    def mouseMoveEvent(self, event):
        if self.isDrawing == True:
            self.endX = event.x()
            self.endY = event.y()
            self.width = self.endX-self.startX
            self.height = self.endY-self.startY
            self.update()
            self.repaint()

    def mouseReleaseEvent(self, event):
        if self.isDrawing == True and event.button() == QtCore.Qt.LeftButton:
            self.isDrawing = False
            self.mapToParent(QPoint(0, 0))
            self.endX = max(min(event.x() - 0, self.geometry().right()), 0)
            self.endY = max(min(event.y() - 0, self.geometry().bottom()), 0)
            print('endX is %d', self.endX)
            print('endY is %d', self.endY)
            self.template = self.image[self.startY:self.endY,self.startX:self.endX]

    def paintEvent(self, event):
        super(Video, self).paintEvent(event)
        p = QPainter()
        p.begin(self)
        if not self.camImage is None:
            p.drawImage(0,0,self.camImage)
        self.mapToParent(QPoint(0, 0))
        if self.isDrawing:
            p.setCompositionMode(QPainter.CompositionMode_SourceOver)
            p.setRenderHint(QPainter.Antialiasing)
            p.setPen(QPen(Qt.blue, 4, Qt.SolidLine))
            p.drawRect(self.startX, self.startY, self.width, self.height)
        p.end()
class Ui_MainWindow(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent) #父类的构造函数

        #self.timer_camera = QtCore.QTimer() #定义定时器，用于控制显示视频的帧率
        #self.cap = cv2.VideoCapture()       #视频流
        #self.CAM_NUM = 0                    #为0时表示视频流来自笔记本内置摄像头

        self.set_ui()                       #初始化程序界面
        self.slot_init()                    #初始化槽函数
        self.status = Template.init #Initialize status of screenshot

        #self.cam = Control(self)

    def set_ui(self):
        self.__layout_main = QtWidgets.QHBoxLayout()           #总布局
        self.__layout_fun_button = QtWidgets.QVBoxLayout()      #按键布局
        self.__layout_temp_button = QtWidgets.QVBoxLayout()  #图像处理按键布局
        #self.__layout_data_show = QtWidgets.QVBoxLayout()       #数据(视频)显示布局sub_slot

        self.button_open_camera = QtWidgets.QPushButton('Enable Camera')  # 建立用于打开摄像头的按键
        self.button_vedio_crop = QtWidgets.QPushButton('Target Crop')
        self.button_close = QtWidgets.QPushButton('Quit')  # 建立用于退出程序的按键
        #self.button_gray = QtWidgets.QPushButton('GRAY')

        self.button_open_camera.setMinimumHeight(50)  # 设置按键大小
        self.button_vedio_crop.setMinimumHeight(50)
        self.button_close.setMinimumHeight(50)
        #self.button_gray.setMinimumHeight(50)

        #self.button_close.move(10, 100)  # 移动按键
        '''信息显示'''
        self.label_show_camera = Video(self)  # 定义显示视频的Label
        self.label_show_camera.setFixedSize(641, 481)  # 给显示视频的Label设置大小为641x481
        '''把按键加入到操作按键布局中'''
        self.__layout_fun_button.addWidget(self.button_open_camera)  # 把打开摄像头的按键放到按键布局中
        self.__layout_fun_button.addWidget(self.button_vedio_crop)  # 把截屏程序的按键放到按键布局中
        self.__layout_fun_button.addWidget(self.button_close)  # 把退出程序的按键放到按键布局中
        #self.__layout_temp_button.addWidget(self.button_gray)
        '''把某些控件加入到总布局中'''
        self.__layout_main.addLayout(self.__layout_fun_button)  # 把按键布局加入到总布局中
        self.__layout_main.addWidget(self.label_show_camera)  # 把用于显示视频的Label加入到总布局中
        #self.__layout_main.addWidget(self.label_show_template)
        #self.__layout_main.addLayout(self.__layout_temp_button)

        self.setLayout(self.__layout_main)  # 到这步才会显示所有控件

    def slot_init(self):
        self.button_open_camera.clicked.connect(self.button_open_camera_clicked)  # 若该按键被点击，则调用button_open_camera_clicked()
        self.button_close.clicked.connect(self.close)  # 若该按键被点击，则调用close()，注意这个close是父类QtWidgets.QWidget自带的，会关闭程序
        self.button_vedio_crop.clicked.connect(self.button_vedio_crop_clicked)  # 若该按键被点击，则调用button_vedio_crop.clicked()

    def button_open_camera_clicked(self):
        self.label_show_camera.setTimer(not self.label_show_camera.onoff)
    def button_vedio_crop_clicked(self):
        if self.status == Template.init:
            self.label_show_camera.isDrawing=True
            self.label_show_camera.timer_camera.stop()
            self.status = Template.choice
            self.button_open_camera.setText('Reactive Cam')
            self.button_vedio_crop.setText('Generate')
        elif self.status == Template.choice:
            reply = QMessageBox.question(self, 'Message', "Are you sure to finish cropping?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.label_show_camera.isDrawing = False
                image = self.label_show_camera.template
                cv2.imwrite('temp.jpg', image)
                self.button_vedio_crop.setText('Target Crop')
                self.status = Template.init
            else:
                self.label_show_camera.isDrawing = True



    #def cam(self):
        #self.cam.show_camera()

if __name__ =='__main__':
    app = QtWidgets.QApplication(sys.argv)  #固定的，表示程序应用
    ui = Ui_MainWindow()                    #实例化Ui_MainWindow
    ui.show()                               #调用ui的show()以显示。同样show()是源于父类QtWidgets.QWidget的
    sys.exit(app.exec_())                   #不加这句，程序界面会一闪而过
