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

class Ui_SizeWeight(QMainWindow):

    def showMain(self, main_menu):
        main_menu.show()

    # Configure webcam to capture the image
    def configCam(self):
        # Insert if statement here to avoid error
        # when the user clicks config button again
        #if not self.thread_running.isOpened():
        #if event.type() == QtCore.QEvent.MouseButtonDblClick:
        self.Work = Work()
        self.Work.start()
        self.Work.eggImg.connect(self.eggImg_slot)


    # Clear image for another image acquisition
    def clearImg(self):
        self.Work.stop()
        self.Work.disconnect()
        self.sizeWeightVid.clear()

           # self.clearImg.isRunning()
           # self.selfCaptured.isRunning()
           # msgbox = QMessageBox()
           # msgbox.setWindowTitle("Warning")
           # msgbox.setText("Invalid capture!!")
           # msgbox.setIcon(QMessageBox.Warning)
           # msgbox.exec_()


    # def displayImage(self, img,window=1):
    #    qformat = QImage.Format_Indexed8

    #    if len(img.shape) == 3:
    #        if (img.shape[2]) == 4:
    #            qformat = QImage.Format_RGBA8888

    #        else:
    #            qformat = QImage.Format_RGBA8888
    #    img = QImage(img, img.shape[1], img.shape[0], qformat)
    #    img = img.rgbSwapped()
    #    self.sizeWeightVid.setPixmap(QPixmap.fromImage(img))
    #    self.sizeWeightVid.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    # Size and Weight video stream container
    def eggImg_slot(self, Image):
        self.sizeWeightVid.setPixmap(QPixmap.fromImage(Image))
        self.sizeWeightVid.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.image_counter = 0

    # Capture the egg image and save it on the folder path
    def captureClicked(self):
        #   self.Work.eggImg.capture(self.eggImg_slot)
        # self.Work.show()
        # self.Work.display()
        # self.sizeWeightVid.capture()
        #self.Work.capture()
        # self.Work.stop()
        # self.sizeWeightVid.clear()
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        ret, frame = cam.read()
        date = datetime.datetime.now()
        filename = r'C:\Users\leste\PycharmProjects\Egg Images\Size and Weight Classification'
        if ret:
            QtWidgets.QApplication.beep()
            cv2.imwrite(os.path.join(filename, "size_weight_egg_img_%s%s%sT%s%s%s.jpg" % (
                date.year, date.month, date.day, date.hour, date.minute, date.second)), frame)

            self.image_counter += 1  # This variable shows how many frames are captured
            self.Work.stop()
            cam.release()
        msgbox = QMessageBox()
        msgbox.setWindowTitle("Image Captured")
        msgbox.setText("Image captured and saved. For another, image just click the clear button and select capture button again")
        msgbox.setIcon(QMessageBox.Information)
        msgbox.exec_()

        # Once the img was captured and saved. The showMessage / dialogBox will appear
        # After capturing the eggs, the image itself will display

            #label = QLabel(self)
            #label.setPixmap(QPixmap(frame))
            #cv2.destroyAllWindows()
            #self.displayImg.show()

        #if filename [0] == "":
        #    return None
        #self.fn = filename[0]

    #def displayImg(self):
        #self.Work.stop()
        #currentFrame = 0
        #self.frame.read()
        #self.frame.show()
        # Read the captured frame from captureClicked button
        #filepath = cam.read()

        #egg_image = QPixmap(self.captureClicked)
        #label = QLabel(self)
        #label.setPixmap(egg_image)
        #self.captureClicked.eggImg.connect(self.eggImg_slot)

    def setupUi(self, SizeWeight, MainMenu):
        SizeWeight.setObjectName("SizeWeight")
        SizeWeight.resize(700, 450)
        self.centralwidget = QtWidgets.QWidget(SizeWeight)
        self.centralwidget.setObjectName("centralwidget")
        self.bgwidget = QtWidgets.QWidget(self.centralwidget)
        self.bgwidget.setGeometry(QtCore.QRect(0, 0, 700, 450))
        self.bgwidget.setStyleSheet("QWidget#bgwidget{\n"
                                    "background-color:rgb(247, 241, 227);}")
        self.bgwidget.setObjectName("bgwidget")
        self.sizeWeightVid = QtWidgets.QLabel(self.bgwidget)
        self.sizeWeightVid.setGeometry(QtCore.QRect(30, 70, 431, 311))
        self.sizeWeightVid.setStyleSheet("background-color:rgb(255, 255, 255);\n"
                                         "color:rgb(85, 85, 85)\n"
                                         "")
        self.sizeWeightVid.setFrameShape(QtWidgets.QFrame.Box)
        self.sizeWeightVid.setLineWidth(0)
        self.sizeWeightVid.setText("")
        self.sizeWeightVid.setObjectName("sizeWeightVid")
        self.configBtn = QtWidgets.QPushButton(self.bgwidget, clicked=lambda: self.configCam())
        self.configBtn.setGeometry(QtCore.QRect(500, 80, 171, 41))
        font = QtGui.QFont()
        font.setFamily("Source Sans Pro")
        font.setPointSize(14)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.configBtn.setFont(font)
        self.configBtn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.configBtn.setStyleSheet("border-radius:10px;\n"
                                     "background-color:qlineargradient(spread:pad, x1:0.216, y1:0.232727, x2:0.847, y2:0.847, stop:0 rgba(255, 177, 66, 255), stop:1 rgba(255, 218, 121, 255));\n"
                                     "color:rgb(255, 255, 255);")
        self.configBtn.setFlat(False)
        self.configBtn.setObjectName("configBtn")
        self.captureBtn = QtWidgets.QPushButton(self.bgwidget, clicked=lambda: self.captureClicked())
        self.captureBtn.setGeometry(QtCore.QRect(500, 140, 171, 41))
        font = QtGui.QFont()
        font.setFamily("Source Sans Pro")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.captureBtn.setFont(font)
        self.captureBtn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.captureBtn.setStyleSheet("border-radius:10px;\n"
                                      "background-color:qlineargradient(spread:pad, x1:0.216, y1:0.232727, x2:0.847, y2:0.847, stop:0 rgba(255, 177, 66, 255), stop:1 rgba(255, 218, 121, 255));\n"
                                      "color:rgb(255, 255, 255);")
        self.captureBtn.setFlat(False)
        self.captureBtn.setObjectName("captureBtn")
        self.clearBtn = QtWidgets.QPushButton(self.bgwidget, clicked=lambda: self.clearImg())
        self.clearBtn.setGeometry(QtCore.QRect(500, 200, 171, 41))
        font = QtGui.QFont()
        font.setFamily("Source Sans Pro")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.clearBtn.setFont(font)
        self.clearBtn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.clearBtn.setStyleSheet("border-radius:10px;\n"
                                    "background-color:qlineargradient(spread:pad, x1:0.216, y1:0.232727, x2:0.847, y2:0.847, stop:0 rgba(255, 177, 66, 255), stop:1 rgba(255, 218, 121, 255));\n"
                                    "color:rgb(255, 255, 255);")
        self.clearBtn.setFlat(False)
        self.clearBtn.setObjectName("clearBtn")
        self.backBtn = QtWidgets.QPushButton(self.bgwidget, clicked=lambda: self.showMain(MainMenu))
        self.backBtn.setGeometry(QtCore.QRect(500, 340, 171, 41))
        font = QtGui.QFont()
        font.setFamily("Source Sans Pro")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.backBtn.setFont(font)
        self.backBtn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.backBtn.setStyleSheet("border-radius:10px;\n"
                                   "background-color:qlineargradient(spread:pad, x1:0.216, y1:0.232727, x2:0.847, y2:0.847, stop:0 rgba(255, 177, 66, 255), stop:1 rgba(255, 218, 121, 255));\n"
                                   "color:rgb(255, 255, 255);")
        self.backBtn.setFlat(False)
        self.backBtn.setObjectName("backBtn")
        self.panelwidget = QtWidgets.QWidget(self.bgwidget)
        self.panelwidget.setGeometry(QtCore.QRect(0, 0, 701, 51))
        self.panelwidget.setStyleSheet("QWidget#panelwidget{\n"
                                       "background-color:qlineargradient(spread:pad, x1:0.216, y1:0.232727, x2:0.847, y2:0.847, stop:0 rgba(255, 121, 63, 255), stop:1 rgba(255, 218, 121, 255));}")
        self.panelwidget.setObjectName("panelwidget")
        self.label_2 = QtWidgets.QLabel(self.panelwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 10, 361, 31))
        font = QtGui.QFont()
        font.setFamily("Source Sans Pro")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("color:rgb(255, 255, 255);")
        self.label_2.setObjectName("label_2")
        self.label_11 = QtWidgets.QLabel(self.bgwidget)
        self.label_11.setGeometry(QtCore.QRect(30, 390, 161, 31))
        font = QtGui.QFont()
        font.setFamily("Source Sans Pro")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.label_11.setFont(font)
        self.label_11.setStyleSheet("")
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.bgwidget)
        self.label_12.setGeometry(QtCore.QRect(200, 390, 261, 31))
        font = QtGui.QFont()
        font.setFamily("Source Sans Pro")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.label_12.setFont(font)
        self.label_12.setStyleSheet("")
        self.label_12.setObjectName("label_12")
        self.sizeLbl = QtWidgets.QLabel(self.bgwidget)
        self.sizeLbl.setGeometry(QtCore.QRect(70, 390, 121, 31))
        font = QtGui.QFont()
        font.setFamily("Source Sans Pro")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.sizeLbl.setFont(font)
        self.sizeLbl.setStyleSheet("")
        self.sizeLbl.setObjectName("sizeLbl")
        self.weightLbl = QtWidgets.QLabel(self.bgwidget)
        self.weightLbl.setGeometry(QtCore.QRect(350, 390, 111, 31))
        font = QtGui.QFont()
        font.setFamily("Source Sans Pro")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.weightLbl.setFont(font)
        self.weightLbl.setStyleSheet("")
        self.weightLbl.setObjectName("weightLbl")
        self.panelwidget.raise_()
        self.sizeWeightVid.raise_()
        self.configBtn.raise_()
        self.captureBtn.raise_()
        self.clearBtn.raise_()
        self.backBtn.raise_()
        self.label_11.raise_()
        self.label_12.raise_()
        self.sizeLbl.raise_()
        self.weightLbl.raise_()
        SizeWeight.setCentralWidget(self.centralwidget)
        #self.statusbar = QtWidgets.QStatusBar(SizeWeight)
        #self.statusbar.setObjectName("statusbar")
        #SizeWeight.setStatusBar(self.statusbar)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        SizeWeight.setSizePolicy(sizePolicy)
        SizeWeight.setMinimumSize(QtCore.QSize(700, 450))
        SizeWeight.setMaximumSize(QtCore.QSize(700, 450))
        self.retranslateUi(SizeWeight)
        QtCore.QMetaObject.connectSlotsByName(SizeWeight)

    def retranslateUi(self, SizeWeight):
        _translate = QtCore.QCoreApplication.translate
        SizeWeight.setWindowTitle(_translate("Size and Weight Classification", "Size and Weight Classification"))
        self.configBtn.setText(_translate("SizeWeight", "Configure Camera"))
        self.captureBtn.setText(_translate("SizeWeight", "Capture"))
        self.clearBtn.setText(_translate("SizeWeight", "Clear"))
        self.backBtn.setText(_translate("SizeWeight", "Back"))
        self.label_2.setText(_translate("SizeWeight", "Size and Weight Classification"))
        #self.label_11.setText(_translate("SizeWeight", "Size:"))
        #self.label_12.setText(_translate("SizeWeight", "Estimated Weight:"))
        #self.sizeLbl.setText(_translate("SizeWeight", "0"))
        #self.weightLbl.setText(_translate("SizeWeight", "0 grams"))
        SizeWeight.setWindowIcon(QIcon('eggsify-logo-small-01.png'))


class Work(QThread):
    eggImg = pyqtSignal(QImage)

    def run(self):
        self.thread_running = True
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while self.thread_running:
            ret, frame = cam.read()
            if ret:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                flip = cv2.flip(img, 1)
                converter = QImage(flip.data, flip.shape[1], flip.shape[0], QImage.Format_RGB888)
                pic = converter.scaled(431, 311, Qt.KeepAspectRatio)
                self.eggImg.emit(pic)

        cam.release()
        cv2.destroyAllWindows()
        return

    def stop(self):
        self.thread_running = False
        self.quit()

    #def capture(self, frame, ret, img):
        # if ret == True:
        #    self.run(frame, 1)
        # self.eggImg_slot(frame, 1)
        # self.fileName(frame, 1)
        # if (self.logic == 2):

        #self.value = self.value + 1
        #date = datetime.datetime.now()
        # fileName=(r"C:\Users\leste\PycharmProjects\Egg Images\image.png", frame)
        # filename = 'C:/Users/leste/PycharmProjects/Egg Images/Egg_Image_%s%s%sT%s%s%s.jpg' % (
        # date.year, date.month, date.day, date.hour, date.minute, date.second)
        # fileName = 'egg_img.jpg',
        # print(fileName)
        # cv2.imwrite('C:\Users\leste\PycharmProjects\Egg Images\%s.png'%(self.value), frame)
        #cv2.imwrite("C:/Users/leste/PycharmProjects/Egg Images/Egg_Image_%s%s%sT%s%s%s.jpg" % (
            #date.year, date.month, date.day, date.hour, date.minute, date.second), img)
        # cv2.imwrite(filename, img)

        # self.logic = 1
        # Dialog box will appear once the img was captured
        # self.setText('Imaged saved. For another image, select capture button again')

    # else:
    #    print('Return not found')





