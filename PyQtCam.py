import sys
from PyQt5.QtWidgets import (QWidget, QToolTip,
    QPushButton, QApplication,QMessageBox,QDesktopWidget)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QCoreApplication

class Cam(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        QToolTip.setFont(QFont('SansSerif', 10))

        self.setToolTip('This is a <b>QWidget</b> widget')

        strat = QPushButton('Start', self)
        strat.setToolTip('This is a <b>Start Button</b> for vedio')
        strat.resize(strat.sizeHint())
        strat.move(50, 50)

        crop = QPushButton('Crop', self)
        crop.setToolTip('This is a <b>Cropper Button</b> for vedio')
        crop.resize(strat.sizeHint())
        crop.move(50, 200)

        qbtn = QPushButton('Quit', self)
        qbtn.clicked.connect(QCoreApplication.instance().quit)
        qbtn.setToolTip('This is a <b>Quit</b> button')
        qbtn.resize(qbtn.sizeHint())
        qbtn.move(50, 350)

        self.resize(640, 480)
        self.center()
        self.setWindowTitle('Tooltips')
        self.show()

    def center(self):

        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def closeEvent(self, event):

        reply = QMessageBox.question(self, 'Message',
                                     "Are you sure to quit?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

if __name__ == '__main__':
  app = QApplication(sys.argv)
  ex = Cam()
  sys.exit(app.exec_())

