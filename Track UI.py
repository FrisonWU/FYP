from PyQt5 import QtCore,QtGui,QtWidgets
from PyQt5.QtGui import QPen, QPainter, QBrush, QBitmap, QPixmap, QImage, qRgb
from PyQt5.QtWidgets import QWidget, QMessageBox, QLabel
from PyQt5.QtCore import QPoint, Qt, QRect
from RealTimeTrack import Project
import sys
import cv2
class CamStatus:
    RGB = 0
    BIN = 1
    BINTRACK=2
    MASSCENTER=3
    STOP = 4

class Model (QLabel):
    def __init__(self,parent=None):
        super().__init__(parent)  # 父类的构造函数
        self.yaw = 0
        self.pitch = 0

class Track(QLabel):
    def __init__(self,parent=None):
        super().__init__(parent) #父类的构造函数
        self.timer_camera = QtCore.QTimer()  # 定义定时器，用于控制显示视频的帧率
        self.cap = cv2.VideoCapture()  # 视频流
        self.CAM_NUM = 0  # 为0时表示视频流来自笔记本内置摄像头
        self.cam_status = CamStatus.RGB
        self.sub_slot()
        self.onoff = False
        self.camImage = None
        self.color_table = [qRgb(i, i, i) for i in range(256)]
        self.threshold = 120

    def setTimer(self, onoff):
        if onoff:
            print('Open')
            flag = self.cap.open(self.CAM_NUM + cv2.CAP_DSHOW)  # Open CAM, cv2.CAP_DSHOW can restart camera without error
            #self.button_open_camera.setText('Disable Camera')k
            self.timer_camera.start(1)
        else:
            print('No')
            self.timer_camera.stop()
            self.cap.release()  # 释放视频流
            self.clear()  # 清空视频显示区域
            #self.button_open_camera.setText('Enable Camera')
        self.onoff = onoff

    def sub_slot(self):
        self.timer_camera.timeout.connect(self.show_camera)  # 若定时器结束，则调用show_camera()

    def show_camera(self):
        flag, self.image = self.cap.read()  # 从视频流中读
        show = cv2.resize(self.image, (640, 480))  # 把读到的帧的大小重新设置为 640x480
        template = cv2.imread('temp_gray.jpg')
        if self.cam_status == CamStatus.RGB:
            show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
            self.camImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        if self.cam_status == CamStatus.BIN:
            show=cv2.cvtColor(show,cv2.COLOR_BGR2GRAY)
            ret, binary = cv2.threshold(show, self.threshold, 255,cv2.THRESH_BINARY)  # set threshold and obtain binary image
            if not ret is False:
                self.camImage = QtGui.QImage(binary.data, binary.shape[1], binary.shape[0], QtGui.QImage.Format_Indexed8)
                self.camImage.setColorTable(self.color_table)
        if self.cam_status == CamStatus.BINTRACK:
            show = cv2.cvtColor(show, cv2.COLOR_BGR2GRAY)
            ret, binary = cv2.threshold(show, self.threshold, 255,cv2.THRESH_BINARY)  # set threshold and obtain binary image
            if not ret is False:
                bintrackImg = Project.Boundbox(binary,template)
                self.camImage = QtGui.QImage(bintrackImg.data, bintrackImg.shape[1], bintrackImg.shape[0],QtGui.QImage.Format_Indexed8)
                self.camImage.setColorTable(self.color_table)
        if self.cam_status == CamStatus.MASSCENTER:
            gray = cv2.cvtColor(show, cv2.COLOR_BGR2GRAY)
            ret, binary = cv2.threshold(gray, self.threshold, 255,cv2.THRESH_BINARY)  # set threshold and obtain binary image
            if not ret is False:
                bintrackImg = Project.Boundbox(binary, template)
                masscenter = Project.MassCenter(bintrackImg,show,20)
                masscenter = cv2.cvtColor(masscenter,cv2.COLOR_BGR2RGB)
                self.camImage = QtGui.QImage(masscenter.data, masscenter.shape[1], masscenter.shape[0], QtGui.QImage.Format_RGB888)


        self.repaint()

    def paintEvent(self, event):
        super(Track, self).paintEvent(event)
        p = QPainter()
        p.begin(self)
        if not self.camImage is None:
            p.drawImage(0, 0, self.camImage)
        self.mapToParent(QPoint(0, 0))
        p.end()

class Ui_TrackWindow(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent) #父类的构造函数

        #self.timer_camera = QtCore.QTimer() #定义定时器，用于控制显示视频的帧率
        #self.cap = cv2.VideoCapture()       #视频流
        #self.CAM_NUM = 0                    #为0时表示视频流来自笔记本内置摄像头
        self.set_ui()                       #初始化程序界面
        self.slot_init()                    #初始化槽函数
        self.image = None
        self.status = CamStatus.STOP
        self.threshold_value = 120

    def set_ui(self):
        self.__layout_main = QtWidgets.QHBoxLayout()  # 总布局

        self.___layout_track_button = QtWidgets.QVBoxLayout()  # Buttons layout for tracking
        self.___layout_model_button = QtWidgets.QVBoxLayout()  # Buttons layout for model

        self.button_track_enable = QtWidgets.QPushButton('Enable Track')
        self.button_bin_vedio = QtWidgets.QPushButton('Binary Vedio')
        self.button_mass_center = QtWidgets.QPushButton('Mass Center')
        self.button_bin_track = QtWidgets.QPushButton('Binary Track')

        self.button_track_enable.setMinimumHeight(50)
        self.button_bin_vedio.setMinimumHeight(50)
        self.button_mass_center.setMinimumHeight(50)
        self.button_bin_track.setMinimumHeight(50)

        '''Create A Slider'''
        self.threshold_slider_copy = QtWidgets.QSlider(Qt.Horizontal)
        self.threshold_slider_copy.setMinimum(0)  # set the minimun value of a slider
        self.threshold_slider_copy.setMaximum(255)  # set the maximum value of a slider
        self.threshold_slider_copy.setSingleStep(5)  # set single step length
        self.threshold_slider_copy.setValue(120)  # set initial value
        self.threshold_slider_copy.setTickPosition(QtWidgets.QSlider.TicksAbove)  # set the tick position of the slider
        self.threshold_slider_copy.setTickInterval(10)  # set tick interval

        '''Create A Dialog to show the value of Slider'''
        self.threshold_dialog_copy = QtWidgets.QLineEdit()

        self.label_show_track = Track(self)
        self.label_show_track.setFixedSize(641, 481)
        self.label_show_model = QtWidgets.QLabel(self)
        self.label_show_model.setFixedSize(641, 481)

        self.___layout_track_button.addWidget(self.button_track_enable)
        self.___layout_track_button.addWidget(self.button_bin_vedio)
        self.___layout_track_button.addWidget(self.button_mass_center)
        self.___layout_track_button.addWidget(self.button_bin_track)
        self.___layout_track_button.addWidget(self.threshold_slider_copy)
        self.___layout_track_button.addWidget(self.threshold_dialog_copy)

        self.__layout_main.addLayout(self.___layout_track_button)
        self.__layout_main.addWidget(self.label_show_track)
        self.__layout_main.addWidget(self.label_show_model)
        self.__layout_main.addLayout(self.___layout_model_button)
        self.setLayout(self.__layout_main)  # 到这步才会显示所有控件


    def slot_init(self):
        self.button_track_enable.clicked.connect(self.button_track_enable_clicked)
        self.button_bin_vedio.clicked.connect(self.button_bin_vedio_clicked)
        self.button_bin_track.clicked.connect(self.button_bin_track_clicked)
        self.button_mass_center.clicked.connect(self.button_mass_center_clicked)
        self.threshold_slider_copy.valueChanged.connect(self.valuechange)
        self.threshold_slider_copy.valueChanged.connect(self.OutDialog)

    def button_track_enable_clicked(self):
        if self.status == CamStatus.STOP:
            self.label_show_track.cam_status = CamStatus.RGB
            self.label_show_track.setTimer(not self.label_show_track.onoff)
            self.status = CamStatus.RGB
        elif not self.status == CamStatus.RGB:
            self.label_show_track.cam_status = CamStatus.RGB
            self.label_show_track.setTimer(self.label_show_track.onoff)
            self.status = CamStatus.RGB
        else:
            self.label_show_track.setTimer(not self.label_show_track.onoff)
            self.status = CamStatus.STOP
        #self.label_show_track.show_camera()
    def button_bin_vedio_clicked(self):
        if self.status == CamStatus.STOP:
            self.label_show_track.cam_status = CamStatus.BIN
            self.label_show_track.setTimer(not self.label_show_track.onoff)
            self.status = CamStatus.BIN
        elif not self.status == CamStatus.BIN:
            self.label_show_track.cam_status = CamStatus.BIN
            self.label_show_track.setTimer(self.label_show_track.onoff)
            self.status = CamStatus.BIN
        else:
            self.label_show_track.setTimer(not self.label_show_track.onoff)
            self.status = CamStatus.STOP
        #self.label_show_track.show_camera()
    def button_bin_track_clicked(self):
        if self.status == CamStatus.STOP:
            self.label_show_track.cam_status = CamStatus.BINTRACK
            self.label_show_track.setTimer(not self.label_show_track.onoff)
            self.status = CamStatus.BINTRACK
        elif not self.status == CamStatus.BINTRACK:
            self.label_show_track.cam_status = CamStatus.BINTRACK
            self.label_show_track.setTimer(self.label_show_track.onoff)
            self.status = CamStatus.BINTRACK
        else:
            self.label_show_track.setTimer(not self.label_show_track.onoff)
            self.status = CamStatus.STOP

    def button_mass_center_clicked(self):
        if self.status == CamStatus.STOP:
            self.label_show_track.cam_status = CamStatus.MASSCENTER
            self.label_show_track.setTimer(not self.label_show_track.onoff)
            self.status = CamStatus.MASSCENTER
        elif not self.status == CamStatus.MASSCENTER:
            self.label_show_track.cam_status = CamStatus.MASSCENTER
            self.label_show_track.setTimer(self.label_show_track.onoff)
            self.status = CamStatus.MASSCENTER
        else:
            self.label_show_track.setTimer(not self.label_show_track.onoff)
            self.status = CamStatus.STOP
    def valuechange(self):
        self.threshold_value = self.threshold_slider_copy.value()
        self.label_show_track.threshold = self.threshold_value
        #self.label_show_track.setTimer(self.label_show_track)
    def OutDialog(self):
        thevalue = str(self.threshold_value)
        self.threshold_dialog_copy.setText(thevalue)


if __name__ =='__main__':
    app = QtWidgets.QApplication(sys.argv)  #固定的，表示程序应用
    ui = Ui_TrackWindow()                    #实例化Ui_MainWindow
    ui.show()                               #调用ui的show()以显示。同样show()是源于父类QtWidgets.QWidget的
    sys.exit(app.exec_())                   #不加这句，程序界面会一闪而过
