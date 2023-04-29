import cv2
import numpy as np
import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from skimage.io import imsave, imread
from PyQt5.uic import loadUi
from PyQt5.QtCore import pyqtSlot, pyqtSignal, Qt, QThread
import datetime
import numpy as np

class Ui_Quality(object):

    def showMain2(self, main_menu):
        main_menu.show()

    # Configure webcam to capture the image
    def configCam2(self):
        #if event.type() == QtCore.QEvent.MouseButtonDblClick:
        self.Work2 = Work2()
        self.Work2.start()
        self.Work2.eggImg2.connect(self.eggImg_slot)

    # Clear image for another image acquisition
    def clearImg(self):
        self.Work2.stop2()
        self.Work2.disconnect()
        self.qualityVid.clear()

    # Quality video stream container
    def eggImg_slot(self, Image):
        self.qualityVid.setPixmap(QPixmap.fromImage(Image))
        self.qualityVid.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.image_counter = 0

    # Capture img and save it on folder path
    def captureClicked2(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        ret, frame = cap.read()
        date = datetime.datetime.now()
        filename = r'C:\Users\leste\PycharmProjects\Egg Images\Quality Classification'
        if ret:
            QtWidgets.QApplication.beep()
            cv2.imwrite(os.path.join(filename, "quality_egg_img_%s%s%sT%s%s%s.jpg" % (
                date.year, date.month, date.day, date.hour, date.minute, date.second)), frame)

            self.image_counter += 1  # This variable shows how many frames are captured
            self.Work2.stop2()
            cap.release()
        msgbox = QMessageBox()
        msgbox.setWindowTitle("Image Captured")
        msgbox.setText("Image captured and saved. For another, image just click the clear button and select capture button again")
        msgbox.setIcon(QMessageBox.Information)
        msgbox.exec_()

    def setupUi(self, Quality, MainMenu):
        Quality.setObjectName("Quality")
        Quality.resize(700, 450)
        self.centralwidget = QtWidgets.QWidget(Quality)
        self.centralwidget.setObjectName("centralwidget")
        self.bgwidget = QtWidgets.QWidget(self.centralwidget)
        self.bgwidget.setGeometry(QtCore.QRect(0, 0, 700, 450))
        self.bgwidget.setStyleSheet("QWidget#bgwidget{\n"
"background-color:rgb(247, 241, 227);}")
        self.bgwidget.setObjectName("bgwidget")
        self.qualityVid = QtWidgets.QLabel(self.bgwidget)
        self.qualityVid.setGeometry(QtCore.QRect(30, 70, 431, 311))
        self.qualityVid.setStyleSheet("background-color:rgb(255, 255, 255);\n"
"color:rgb(85, 85, 85)\n"
"")
        #self.qualityVid.setPixmap(QtGui.QPixmap(self.eggImg_slot2)) # setPixmap
        self.qualityVid.setFrameShape(QtWidgets.QFrame.Box)
        self.qualityVid.setLineWidth(0)
        self.qualityVid.setText("")
        self.qualityVid.setObjectName("qualityVid")
        self.configBtn1 = QtWidgets.QPushButton(self.bgwidget, clicked=lambda: self.configCam2())
        self.configBtn1.setGeometry(QtCore.QRect(500, 80, 171, 41))
        font = QtGui.QFont()
        font.setFamily("Source Sans Pro")
        font.setPointSize(14)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.configBtn1.setFont(font)
        self.configBtn1.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.configBtn1.setStyleSheet("border-radius:10px;\n"
"background-color:qlineargradient(spread:pad, x1:0.216, y1:0.232727, x2:0.847, y2:0.847, stop:0 rgba(255, 177, 66, 255), stop:1 rgba(255, 218, 121, 255));\n"
"color:rgb(255, 255, 255);")
        self.configBtn1.setFlat(False)
        self.configBtn1.setObjectName("configBtn1")
        self.captureBtn1 = QtWidgets.QPushButton(self.bgwidget, clicked=lambda: self.captureClicked2())
        self.captureBtn1.setGeometry(QtCore.QRect(500, 140, 171, 41))
        font = QtGui.QFont()
        font.setFamily("Source Sans Pro")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.captureBtn1.setFont(font)
        self.captureBtn1.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.captureBtn1.setStyleSheet("border-radius:10px;\n"
"background-color:qlineargradient(spread:pad, x1:0.216, y1:0.232727, x2:0.847, y2:0.847, stop:0 rgba(255, 177, 66, 255), stop:1 rgba(255, 218, 121, 255));\n"
"color:rgb(255, 255, 255);")
        self.captureBtn1.setFlat(False)
        self.captureBtn1.setObjectName("captureBtn1")
        self.clearBtn1 = QtWidgets.QPushButton(self.bgwidget, clicked=lambda: self.clearImg())
        self.clearBtn1.setGeometry(QtCore.QRect(500, 200, 171, 41))
        font = QtGui.QFont()
        font.setFamily("Source Sans Pro")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.clearBtn1.setFont(font)
        self.clearBtn1.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.clearBtn1.setStyleSheet("border-radius:10px;\n"
"background-color:qlineargradient(spread:pad, x1:0.216, y1:0.232727, x2:0.847, y2:0.847, stop:0 rgba(255, 177, 66, 255), stop:1 rgba(255, 218, 121, 255));\n"
"color:rgb(255, 255, 255);")
        self.clearBtn1.setFlat(False)
        self.clearBtn1.setObjectName("clearBtn1")
        self.backBtn1 = QtWidgets.QPushButton(self.bgwidget, clicked=lambda: self.showMain2(MainMenu))
        self.backBtn1.setGeometry(QtCore.QRect(500, 340, 171, 41))
        font = QtGui.QFont()
        font.setFamily("Source Sans Pro")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.backBtn1.setFont(font)
        self.backBtn1.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.backBtn1.setStyleSheet("border-radius:10px;\n"
"background-color:qlineargradient(spread:pad, x1:0.216, y1:0.232727, x2:0.847, y2:0.847, stop:0 rgba(255, 177, 66, 255), stop:1 rgba(255, 218, 121, 255));\n"
"color:rgb(255, 255, 255);")
        self.backBtn1.setFlat(False)
        self.backBtn1.setObjectName("backBtn1")
        self.panelwidget = QtWidgets.QWidget(self.bgwidget)
        self.panelwidget.setGeometry(QtCore.QRect(0, 0, 701, 51))
        self.panelwidget.setStyleSheet("QWidget#panelwidget{\n"
"background-color:qlineargradient(spread:pad, x1:0.216, y1:0.232727, x2:0.847, y2:0.847, stop:0 rgba(255, 121, 63, 255), stop:1 rgba(255, 218, 121, 255));}")
        self.panelwidget.setObjectName("panelwidget")
        self.label_2 = QtWidgets.QLabel(self.panelwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 10, 231, 31))
        font = QtGui.QFont()
        font.setFamily("Source Sans Pro")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("color:rgb(255, 255, 255);")
        self.label_2.setObjectName("label_2")
        self.qualityLbl = QtWidgets.QLabel(self.bgwidget)
        self.qualityLbl.setGeometry(QtCore.QRect(30, 390, 161, 31))
        font = QtGui.QFont()
        font.setFamily("Source Sans Pro")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.qualityLbl.setFont(font)
        self.qualityLbl.setStyleSheet("")
        self.qualityLbl.setObjectName("qualityLbl")
        self.typeOfEgg = QtWidgets.QLabel(self.bgwidget)
        self.typeOfEgg.setGeometry(QtCore.QRect(120, 390, 161, 31))
        font = QtGui.QFont()
        font.setFamily("Source Sans Pro")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.typeOfEgg.setFont(font)
        self.typeOfEgg.setStyleSheet("")
        self.typeOfEgg.setObjectName("typeOfEgg")
        self.panelwidget.raise_()
        self.qualityVid.raise_()
        self.configBtn1.raise_()
        self.captureBtn1.raise_()
        self.clearBtn1.raise_()
        self.backBtn1.raise_()
        self.qualityLbl.raise_()
        self.typeOfEgg.raise_()
        Quality.setCentralWidget(self.centralwidget)
        #self.statusbar = QtWidgets.QStatusBar(Quality)
        #self.statusbar.setObjectName("statusbar")
        #Quality.setStatusBar(self.statusbar)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        Quality.setSizePolicy(sizePolicy)
        #self.setFixedSize(QtCore.QSize(700, 450))
        Quality.setMinimumSize(QtCore.QSize(700, 450))
        Quality.setMaximumSize(QtCore.QSize(700, 450))
        self.retranslateUi(Quality)
        QtCore.QMetaObject.connectSlotsByName(Quality)

    def retranslateUi(self, Quality):
        _translate = QtCore.QCoreApplication.translate
        Quality.setWindowTitle(_translate("Quality Classification", "Quality Classification"))
        self.configBtn1.setText(_translate("Quality", "Configure Camera"))
        self.captureBtn1.setText(_translate("Quality", "Capture"))
        self.clearBtn1.setText(_translate("Quality", "Clear"))
        self.backBtn1.setText(_translate("Quality", "Back"))
        self.label_2.setText(_translate("Quality", "Quality Classification"))
        #self.qualityLbl.setText(_translate("Quality", "Quality:"))
        #self.typeOfEgg.setText(_translate("Quality", "0"))
        Quality.setWindowIcon(QIcon('eggsify-logo-small-01.png'))

class Work2(QThread):
    eggImg2 = pyqtSignal(QImage)

    #def __init__(self):
     #   super().__init__()
      #  self.thread_running = True

        # Create QLabel that holds the video stream
       # self.qualityVid = QLabel(self)
        # self.configBtn1.clicked.connect(self.configCam2)
        #self.captureBtn1.clicked.connect(self.captureClicked2())

    def run(self):
        self.thread_running2 = True
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while self.thread_running2:
            ret, frame = cap.read()
            if ret:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                flip = cv2.flip(img, 1)
                converter = QImage(flip.data, flip.shape[1], flip.shape[0], QImage.Format_RGB888)
                pic = converter.scaled(431, 311, Qt.KeepAspectRatio)
                self.eggImg2.emit(pic)

        cap.release()
        cv2.destroyAllWindows()
        return

    def stop2(self):
        self.thread_running2 = False
        self.quit()






