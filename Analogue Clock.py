from PyQt5 import QtCore,QtGui,QtWidgets
from PyQt5.QtGui import QPen, QPainter, QBrush, QBitmap, QPixmap, QImage, qRgb
from PyQt5.QtWidgets import QWidget, QMessageBox, QLabel
from PyQt5.QtCore import QPoint, Qt, QRect
import sys
import math
import numpy
import cv2
class flag:
    yes=1
    no = 0


class Yaw_Pitch(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)  # 父类的构造函数
        self.yawvalue = 0
        self.pitchvalue =0
        self.yawmax = 60
        self.pitchmax = 45
        self.diameter = 400
        self.radius = 200
        self.centerx = 250
        self.centery =250
        self.Pointx = 250
        self.Pointy = 250

    def location(self):
        yawrad = (2*math.pi*self.yawvalue/360)*(90/self.yawmax)
        pitchrad= (2*math.pi*self.pitchvalue/360)*(90/self.pitchmax)
        #deltax = self.Pointx-self.centerx
        #deltay = self.Pointy-self.centery
        #radius_calc = math.sqrt(pow(deltax, 2) + pow(deltay, 2))
        deltax = math.sin(yawrad)
        deltay = math.sin(pitchrad)
        if not pow(deltax,2)+pow(deltay,2) > 1:
            self.Pointx = (math.sin(yawrad)*self.radius) + self.centerx
            self.Pointy = self.centery-(math.sin(pitchrad)*self.radius)
        else:
            theta = math.atan(deltay/deltax)
            if deltax < 0:
                self.Pointx = self.centerx - self.radius*math.cos(theta)
                self.Pointy = self.centery + self.radius * math.sin(theta)
            if deltax >= 0:
                self.Pointx = self.centerx + self.radius*math.cos(theta)
                self.Pointy = self.centery - self.radius * math.sin(theta)
        self.repaint()


    def paintEvent(self, event):
        point = QPoint(self.Pointx,self.Pointy)
        p = QPainter()
        p.begin(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(Qt.blue,8,Qt.SolidLine))
        p.drawEllipse(50, 50, 400, 400)
        p.setPen(QPen(Qt.red,16,Qt.SolidLine))
        p.drawPoint(point)
        p.end()
class Analogue_Clock(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)  # 父类的构造函数

        # self.timer_camera = QtCore.QTimer() #定义定时器，用于控制显示视频的帧率
        # self.cap = cv2.VideoCapture()       #视频流
        # self.CAM_NUM = 0                    #为0时表示视频流来自笔记本内置摄像头
        self.set_ui()  # 初始化程序界面
        self.slot_init()  # 初始化槽函数
        self.yaw = 0
        self.pitch = 0

    def set_ui(self):
        self.__layout_main = QtWidgets.QHBoxLayout()  # 总布局
        self.___layout_clock_button = QtWidgets.QVBoxLayout()

        '''Create A Slider'''
        self.yaw_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.yaw_slider.setMinimum(-60)  # set the minimun value of a slider
        self.yaw_slider.setMaximum(60)  # set the maximum value of a slider
        self.yaw_slider.setSingleStep(5)  # set single step length
        self.yaw_slider.setValue(0)  # set initial value
        self.yaw_slider.setTickPosition(QtWidgets.QSlider.TicksAbove)  # set the tick position of the slider
        self.yaw_slider.setTickInterval(10)  # set tick interval

        self.pitch_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.pitch_slider.setMinimum(-45)  # set the minimun value of a slider
        self.pitch_slider.setMaximum(45)  # set the maximum value of a slider
        self.pitch_slider.setSingleStep(5)  # set single step length
        self.pitch_slider.setValue(0)  # set initial value
        self.pitch_slider.setTickPosition(QtWidgets.QSlider.TicksAbove)  # set the tick position of the slider
        self.pitch_slider.setTickInterval(10)  # set tick interval

        '''Create A Dialog to show the value of Slider'''
        self.yaw_dialog = QtWidgets.QLineEdit()
        self.pitch_dialog = QtWidgets.QLineEdit()

        self.label_show_clock = Yaw_Pitch(self)
        self.label_show_clock.setFixedSize(641, 481)

        self.___layout_clock_button.addWidget(self.yaw_slider)
        self.___layout_clock_button.addWidget(self.yaw_dialog)
        self.___layout_clock_button.addWidget(self.pitch_slider)
        self.___layout_clock_button.addWidget(self.pitch_dialog)

        self.__layout_main.addLayout(self.___layout_clock_button)
        self.__layout_main.addWidget(self.label_show_clock)
        self.setLayout(self.__layout_main)  # 到这步才会显示所有控件

    def slot_init(self):
        self.yaw_slider.valueChanged.connect(self.Yawvaluechange)
        self.yaw_slider.valueChanged.connect(self.YawDialog)
        self.pitch_slider.valueChanged.connect(self.Pitchvaluechange)
        self.pitch_slider.valueChanged.connect(self.PitchDialog)

    def Yawvaluechange(self):
        self.yaw = self.yaw_slider.value()
        self.label_show_clock.yawvalue=self.yaw
        self.label_show_clock.location()
    def YawDialog(self):
        yawvalue = str(self.yaw)
        self.yaw_dialog.setText(yawvalue)

    def Pitchvaluechange(self):
        self.pitch = self.pitch_slider.value()
        self.label_show_clock.pitchvalue = self.pitch
        self.label_show_clock.location()

    def PitchDialog(self):
        pitchvalue = str(self.pitch)
        self.pitch_dialog.setText(pitchvalue)

if __name__ =='__main__':
    app = QtWidgets.QApplication(sys.argv)  #固定的，表示程序应用
    ui = Analogue_Clock()                    #实例化Ui_MainWindow
    ui.show()                               #调用ui的show()以显示。同样show()是源于父类QtWidgets.QWidget的
    sys.exit(app.exec_())                   #不加这句，程序界面会一闪而过



