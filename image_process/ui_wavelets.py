# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI/Ui_wavelets.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Wavelets(object):
    def setupUi(self, Wavelets):
        Wavelets.setObjectName("Wavelets")
        Wavelets.resize(990, 477)
        self.pbn_FWT = QtWidgets.QPushButton(Wavelets)
        self.pbn_FWT.setGeometry(QtCore.QRect(670, 290, 141, 31))
        self.pbn_FWT.setObjectName("pbn_FWT")
        self.lineEdit_level = QtWidgets.QLineEdit(Wavelets)
        self.lineEdit_level.setGeometry(QtCore.QRect(900, 320, 41, 21))
        self.lineEdit_level.setObjectName("lineEdit_level")
        self.label = QtWidgets.QLabel(Wavelets)
        self.label.setGeometry(QtCore.QRect(830, 320, 31, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_inputImg = QtWidgets.QLabel(Wavelets)
        self.label_inputImg.setGeometry(QtCore.QRect(20, 100, 300, 300))
        self.label_inputImg.setStyleSheet("QLabel {\n"
"    font-family: \"Microsoft YaHei\";\n"
"    font-size: 14px;\n"
"    color: #BDC8E2;\n"
"    background-color:#b4b4b4;\n"
"}")
        self.label_inputImg.setAlignment(QtCore.Qt.AlignCenter)
        self.label_inputImg.setObjectName("label_inputImg")
        self.label_2 = QtWidgets.QLabel(Wavelets)
        self.label_2.setGeometry(QtCore.QRect(140, 430, 81, 21))
        self.label_2.setObjectName("label_2")
        self.pbn_loadImg = QtWidgets.QPushButton(Wavelets)
        self.pbn_loadImg.setGeometry(QtCore.QRect(20, 40, 71, 31))
        self.pbn_loadImg.setObjectName("pbn_loadImg")
        self.label_outputImg = QtWidgets.QLabel(Wavelets)
        self.label_outputImg.setGeometry(QtCore.QRect(340, 100, 300, 300))
        self.label_outputImg.setStyleSheet("QLabel {\n"
"    font-family: \"Microsoft YaHei\";\n"
"    font-size: 14px;\n"
"    color: #BDC8E2;\n"
"    background-color:#b4b4b4;\n"
"}")
        self.label_outputImg.setAlignment(QtCore.Qt.AlignCenter)
        self.label_outputImg.setObjectName("label_outputImg")
        self.label_3 = QtWidgets.QLabel(Wavelets)
        self.label_3.setGeometry(QtCore.QRect(470, 420, 71, 21))
        self.label_3.setObjectName("label_3")
        self.pbn_IFWT = QtWidgets.QPushButton(Wavelets)
        self.pbn_IFWT.setGeometry(QtCore.QRect(670, 330, 141, 31))
        self.pbn_IFWT.setObjectName("pbn_IFWT")
        self.pbn_pyramid = QtWidgets.QPushButton(Wavelets)
        self.pbn_pyramid.setGeometry(QtCore.QRect(670, 170, 141, 31))
        self.pbn_pyramid.setObjectName("pbn_pyramid")
        self.label_4 = QtWidgets.QLabel(Wavelets)
        self.label_4.setGeometry(QtCore.QRect(830, 160, 31, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.lineEdit_level_pyramid = QtWidgets.QLineEdit(Wavelets)
        self.lineEdit_level_pyramid.setGeometry(QtCore.QRect(900, 160, 41, 21))
        self.lineEdit_level_pyramid.setObjectName("lineEdit_level_pyramid")
        self.lineEdit_kernel_size = QtWidgets.QLineEdit(Wavelets)
        self.lineEdit_kernel_size.setGeometry(QtCore.QRect(900, 190, 41, 21))
        self.lineEdit_kernel_size.setObjectName("lineEdit_kernel_size")
        self.label_5 = QtWidgets.QLabel(Wavelets)
        self.label_5.setGeometry(QtCore.QRect(830, 190, 51, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")

        self.retranslateUi(Wavelets)
        QtCore.QMetaObject.connectSlotsByName(Wavelets)

    def retranslateUi(self, Wavelets):
        _translate = QtCore.QCoreApplication.translate
        Wavelets.setWindowTitle(_translate("Wavelets", "Form"))
        self.pbn_FWT.setText(_translate("Wavelets", "哈尔FWT"))
        self.lineEdit_level.setText(_translate("Wavelets", "3"))
        self.label.setText(_translate("Wavelets", "层:"))
        self.label_inputImg.setText(_translate("Wavelets", "原图"))
        self.label_2.setText(_translate("Wavelets", "待处理图片"))
        self.pbn_loadImg.setText(_translate("Wavelets", "载入图片"))
        self.label_outputImg.setText(_translate("Wavelets", "处理后图片"))
        self.label_3.setText(_translate("Wavelets", "处理后图片"))
        self.pbn_IFWT.setText(_translate("Wavelets", "哈尔IFWT"))
        self.pbn_pyramid.setText(_translate("Wavelets", "高斯-拉普拉斯金字塔"))
        self.label_4.setText(_translate("Wavelets", "层:"))
        self.lineEdit_level_pyramid.setText(_translate("Wavelets", "4"))
        self.lineEdit_kernel_size.setText(_translate("Wavelets", "5"))
        self.label_5.setText(_translate("Wavelets", "核大小:"))
