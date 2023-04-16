import cv2
import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUi
from sizeWeight import *
from qualityUI import *

class Ui_MainMenu(QWidget):

    def openWindow(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_SizeWeight()
        self.ui.setupUi(self.window, MainMenu)
        MainMenu.hide()
        self.window.show()

        # Quality button
    def openWindow2(self):
        #Quality = QtWidgets.QMainWindow()
        self.window2 = QtWidgets.QMainWindow()
        self.ui = Ui_Quality()
        self.ui.setupUi(self.window2, MainMenu)
        MainMenu.hide()
        self.window2.show()

    def setupUi(self, MainMenu):
        MainMenu.setObjectName("MainMenu")
        MainMenu.setWindowModality(QtCore.Qt.NonModal)
        MainMenu.resize(700, 450)
        MainMenu.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.centralwidget = QtWidgets.QWidget(MainMenu)
        self.centralwidget.setObjectName("centralwidget")
        self.panelwidget = QtWidgets.QWidget(self.centralwidget)
        self.panelwidget.setGeometry(QtCore.QRect(0, 0, 700, 450))
        self.panelwidget.setStyleSheet("QWidget#bgwidget{\n"
"background-color:qlineargradient(spread:pad, x1:0.216, y1:0.232727, x2:1, y2:1, stop:0 rgba(255, 219, 88, 255), stop:1 rgba(255, 255, 255, 255));}")
        self.panelwidget.setObjectName("panelwidget")
        self.bgwidget = QtWidgets.QWidget(self.panelwidget)
        self.bgwidget.setGeometry(QtCore.QRect(0, 0, 700, 450))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        #sizePolicy.setHeightForWidth(self.bgwidget.sizePolicy().hasHeightForWidth())
        self.bgwidget.setSizePolicy(sizePolicy)
        self.bgwidget.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.bgwidget.setStyleSheet("QWidget#bgwidget{\n"
"background-color:qlineargradient(spread:pad, x1:0.216, y1:0.232727, x2:0.847, y2:0.847, stop:0 rgba(255, 121, 63, 255), stop:1 rgba(255, 218, 121, 255));}")
        self.bgwidget.setObjectName("bgwidget")
        self.label_3 = QtWidgets.QLabel(self.bgwidget)
        self.label_3.setGeometry(QtCore.QRect(220, 0, 251, 251))
        font = QtGui.QFont()
        font.setFamily("Source Sans Pro")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("background-image: url(:/newPrefix/Eggsify-logo-01.png);")
        self.label_3.setText("")
        self.label_3.setPixmap(QtGui.QPixmap("D:/Pictures/Graphic Design/Eggsify/Eggsify-logo-small-01.png"))
        self.label_3.setScaledContents(True)
        self.label_3.setObjectName("label_3")
        self.sizeWeightBtn = QtWidgets.QPushButton(self.bgwidget, clicked=lambda: self.openWindow())
        self.sizeWeightBtn.setGeometry(QtCore.QRect(210, 250, 271, 51))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(13)
        font.setBold(False)
        font.setWeight(50)
        self.sizeWeightBtn.setFont(font)
        self.sizeWeightBtn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.sizeWeightBtn.setStyleSheet("border-radius:20px;\n""background-color:rgb(249, 249, 249)\n""")
        self.sizeWeightBtn.setFlat(False)
        self.sizeWeightBtn.setObjectName("sizeWeightBtn")
        self.qualityBtn = QtWidgets.QPushButton(self.bgwidget, clicked=lambda: self.openWindow2())
        self.qualityBtn.setGeometry(QtCore.QRect(210, 310, 271, 51))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(13)
        font.setBold(False)
        font.setWeight(50)
        self.qualityBtn.setFont(font)
        self.qualityBtn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.qualityBtn.setStyleSheet("border-radius:20px;\n"
"background-color:rgb(249, 249, 249)\n"
"")
        self.qualityBtn.setObjectName("qualityBtn")
        MainMenu.setCentralWidget(self.centralwidget)
        MainMenu.setSizePolicy(sizePolicy)
        #self.setFixedSize(QtCore.QSize(700, 450))
        MainMenu.setMinimumSize(QtCore.QSize(700, 450))
        MainMenu.setMaximumSize(QtCore.QSize(700, 450))
        self.retranslateUi(MainMenu)
        QtCore.QMetaObject.connectSlotsByName(MainMenu)

    def retranslateUi(self, MainMenu):
        _translate = QtCore.QCoreApplication.translate
        MainMenu.setWindowTitle(_translate("Main Menu", "Main Menu"))
        self.sizeWeightBtn.setText(_translate("Main Menu", "Size and Weight Classification "))
        self.qualityBtn.setText(_translate("Main Menu", "Quality Classification "))
        MainMenu.setWindowIcon(QIcon('eggsify-logo-small-01.png'))

# Initialize the app
app = QApplication(sys.argv)
MainMenu = QtWidgets.QMainWindow()
ui = Ui_MainMenu()
ui.setupUi(MainMenu)
MainMenu.show()
sys.exit(app.exec_())
