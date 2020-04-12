from PyQt5 import QtCore,QtGui,QtWidgets
from PyQt5.QtGui import QPen, QPainter, QBrush, QBitmap, QPixmap, QImage
from PyQt5.QtWidgets import QWidget, QMessageBox, QLabel
from PyQt5.QtCore import QPoint, Qt, QRect
import sys
import cv2
class ScreenshotStatus:
    init = 0
    choice =1
    ok = 2
    drafting =3

class Control(QLabel):
    def __init__(self,parent=None):
        super().__init__(parent) #父类的构造函数
        self.pix = QPixmap()  # 实例化一个 QPixmap 对象
        self.startX = 0  # 截屏初始X坐标
        self.startY = 0  # 截屏初始Y坐标
        self.lastX = 0
        self.lastY = 0
        self.endX = 0  # 截屏结束X坐标
        self.endY = 0  # 截屏结束Y坐标
        self.isDrawing = False
    def sub_slot(self):
        self.timer_camera.timeout.connect(self.show_camera())  # 若定时器结束，则调用show_camera()
    def show_temp(self):
        self.temp = cv2.imread('temp.jpg')  # 从视频流中读
        temp = cv2.resize(self.temp, (320, 240))  # 把读到的帧的大小重新设置为 640x480
        # temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        tempImage = QtGui.QImage(temp.data, temp.shape[1], temp.shape[0],QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.label_show_template.setPixmap(QtGui.QPixmap.fromImage(tempImage))  # 往显示视频的Label里 显示QImage

    def show_camera(self):
        flag, self.image = self.cap.read()  # 从视频流中读
        show = cv2.resize(self.image, (640, 480))  # 把读到的帧的大小重新设置为 640x480
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage

    def mousePressEvent(self, event):
        if self.isDrawing and event.button() == Qt.LeftButton:
            lt = self.label_show_camera.mapToParent(QPoint(0, 0))
            self.startX = min(max(event.x() - lt.x(), 0), self.label_show_camera.geometry().right())
            self.startY = min(max(event.y() - lt.y(), 0), self.label_show_camera.geometry().bottom())
            print('startX is %d', self.startX)
            print('startY is %d', self.startY)

    def mouseMoveEvent(self, event):
        if self.isDrawing == True:
            self.endX = event.x()
            self.endY = event.y()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.isDrawing == True and event.button() == QtCore.Qt.LeftButton:
            lt = self.label_show_camera.mapToParent(QPoint(0, 0))
            self.endX = max(min(event.x() - lt.x(), self.label_show_camera.geometry().right()), 0)
            self.endY = max(min(event.y() - lt.y(), self.label_show_camera.geometry().bottom()), 0)
            print('endX is %d', self.endX)
            print('endY is %d', self.endY)
            msg = QtWidgets.QMessageBox.warning(self, 'Note', "Crop Done", buttons=QtWidgets.QMessageBox.Ok)

    def paintEvent(self, event):
        # super().paintEvent(event)
        if self.isDrawing:
            print('Painting Start')
            p = QPainter(self)
            p.setCompositionMode(QPainter.CompositionMode_SourceOver)
            p.setRenderHint(QPainter.Antialiasing)

            p.setPen(QPen(Qt.blue, 4, Qt.SolidLine))
            p.drawRect(self.startX, self.startY, self.endX, self.endY)

class Ui_MainWindow(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent) #父类的构造函数

        self.timer_camera = QtCore.QTimer() #定义定时器，用于控制显示视频的帧率
        self.cap = cv2.VideoCapture()       #视频流
        self.CAM_NUM = 0                    #为0时表示视频流来自笔记本内置摄像头

        self.set_ui()                       #初始化程序界面
        self.slot_init()                    #初始化槽函数
        self.status = ScreenshotStatus.init #Initialize status of screenshot

        self.cam = Control(self)
    '''程序界面布局'''
    def set_ui(self):
        self.__layout_main = QtWidgets.QHBoxLayout()           #总布局
        self.__layout_fun_button = QtWidgets.QVBoxLayout()      #按键布局
        self.__layout_process_button = QtWidgets.QVBoxLayout()  #图像处理按键布局
        self.__layout_data_show = QtWidgets.QVBoxLayout()       #数据(视频)显示布局

        self.button_open_camera = QtWidgets.QPushButton('Enable Camera') #建立用于打开摄像头的按键
        self.button_vedio_crop = QtWidgets.QPushButton('Target Crop')
        self.button_close = QtWidgets.QPushButton('Quit')           #建立用于退出程序的按键

        self.button_convert_gray = QtWidgets.QPushButton('GrayScale') #灰度图

        self.button_open_camera.setMinimumHeight(50)                #设置按键大小
        self.button_vedio_crop.setMinimumHeight(50)
        self.button_close.setMinimumHeight(50)

        self.button_close.move(10,100)                      #移动按键
        '''信息显示'''
        self.label_show_camera = QtWidgets.QLabel(self) #定义显示视频的Label
        self.label_show_camera.setFixedSize(641,481)    #给显示视频的Label设置大小为641x481
        self.label_show_template = QtWidgets.QLabel(self)
        self.label_show_template.setFixedSize(321,241)

        '''把按键加入到操作按键布局中'''
        self.__layout_fun_button.addWidget(self.button_open_camera) #把打开摄像头的按键放到按键布局中
        self.__layout_fun_button.addWidget(self.button_vedio_crop)       #把截屏程序的按键放到按键布局中
        self.__layout_fun_button.addWidget(self.button_close)  # 把退出程序的按键放到按键布局中
        self.__layout_fun_button.addWidget(self.button_convert_gray)
        '''把按键加入到图像操作布局中'''
        #self.__layout_process_button.addWidget(self.button_convert_gray)#专灰度图按钮
        '''把某些控件加入到总布局中'''
        self.__layout_main.addLayout(self.__layout_fun_button)      #把按键布局加入到总布局中
        self.__layout_main.addWidget(self.label_show_camera)        #把用于显示视频的Label加入到总布局中
        self.__layout_main.addWidget(self.label_show_template)
        #self.__layout_main.addWidget(self.__layout_process_button)
        '''总布局布置好后就可以把总布局作为参数传入下面函数'''
        self.setLayout(self.__layout_main) #到这步才会显示所有控件

    '''初始化所有槽函数'''
    def slot_init(self):
        self.button_open_camera.clicked.connect(self.button_open_camera_clicked)    #若该按键被点击，则调用button_open_camera_clicked()
        self.button_open_camera.clicked.connect(self.cam)  # 若该按键被点击，则调用button_open_camera_clicked()
        self.button_vedio_crop.clicked.connect(self.button_vedio_crop_clicked)  # 若该按键被点击，则调用button_vedio_crop.clicked()
        #self.timer_camera.timeout.connect(self.cam) #若定时器结束，则调用show_camera()
        #self.button_vedio_crop.clicked.connect(Control.show_temp)  # 若定时器结束，则调用show_camera()
        self.button_close.clicked.connect(self.close)#若该按键被点击，则调用close()，注意这个close是父类QtWidgets.QWidget自带的，会关闭程序
        self.button_convert_gray.clicked.connect(self.button_convert_gray_clicked)

    '''槽函数之一'''
    def button_open_camera_clicked(self):
        if self.timer_camera.isActive() == False:   #若定时器未启动
            flag = self.cap.open(self.CAM_NUM+cv2.CAP_DSHOW) #Open CAM, cv2.CAP_DSHOW can restart camera without error
            if flag == False:       #flag表示open()成不成功
                msg = QtWidgets.QMessageBox.warning(self,'warning',"请检查相机于电脑是否连接正确",buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(30)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
                self.button_open_camera.setText('Disable Camera')
        else:
            self.timer_camera.stop()  #关闭定时器
            self.cap.release()        #释放视频流
            self.label_show_camera.clear()  #清空视频显示区域
            self.button_open_camera.setText('Enable Camera')

    def cam(self):
        self.cam.show()
        self.cam.exec()
    def button_vedio_crop_clicked(self):
        if self.timer_camera.isActive() == False and self.status != ScreenshotStatus.choice and self.status != ScreenshotStatus.ok:
            msg = QtWidgets.QMessageBox.warning(self, 'warning', "Camera was not active!",buttons=QtWidgets.QMessageBox.Ok)
            self.timer_camera.stop()
            self.cap.release()
            #self.button_open_camera.setText('Reactive Cam')
            #self.button_vedio_crop.setText('Finish')
            #if self.status == ScreenshotStatus.init:
        elif self.status == ScreenshotStatus.init: #开始截屏
            msg = QtWidgets.QMessageBox.warning(self, 'Note', "Please Crop the Image",buttons=QtWidgets.QMessageBox.Ok)
            self.timer_camera.stop()
            self.cap.release()

            self.button_open_camera.setText('Reactive Cam')
            #cv2.imwrite('temp', self.show_camera.showIm age)
            self.button_vedio_crop.setText('Finish')
            self.isDrawing = True
            self.status = ScreenshotStatus.choice

        elif self.status == ScreenshotStatus.choice:
            reply = QMessageBox.question(self, 'Message',"Are you sure to finish cropping?", QMessageBox.Yes |QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.isDrawing = False
                self.status = ScreenshotStatus.ok
                self.button_vedio_crop.setText('Generate')

        elif self.status == ScreenshotStatus.ok:
            msg = QtWidgets.QMessageBox.warning(self, 'Note', "Template Image Created", buttons=QtWidgets.QMessageBox.Ok)
            image = self.image[self.startY:self.endY,self.startX:self.endX]
            cv2.imwrite('temp.jpg', image)
            self.button_vedio_crop.setText('Target Crop')
            self.status = ScreenshotStatus.init

    def button_convert_gray_clicked(self):
        if self.temp is None:
            msg = QtWidgets.QMessageBox.warning(self, 'Warning', "Please Crop Target!", buttons=QtWidgets.QMessageBox.Ok)
        else:
            self.temp = cv2.cvtColor(self.temp,cv2.COLOR_BGR2GRAY)




if __name__ =='__main__':
    app = QtWidgets.QApplication(sys.argv)  #固定的，表示程序应用
    ui = Ui_MainWindow()                    #实例化Ui_MainWindow
    ui.show()                               #调用ui的show()以显示。同样show()是源于父类QtWidgets.QWidget的
    sys.exit(app.exec_())                   #不加这句，程序界面会一闪而过