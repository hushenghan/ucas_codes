import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPalette, QBrush, QPixmap, QImage
from PyQt5 import QtCore, QtGui, QtWidgets
import os
from ui_wavelets import Ui_Wavelets
import cv2
import numpy as np
from utils import *
pi = np.pi


class UiWavelets(QtWidgets.QWidget, Ui_Wavelets):
    def __init__(self):
        super(UiWavelets,self).__init__()
        self.setupUi(self)
        self.srcImg = None
        self.srcImg_float = None
        self.grayImg = None
        self.wvts = None
        self.level = 0
        self.pyramid = None

        self.pbn_loadImg.clicked.connect(self.load_img)
        self.pbn_FWT.clicked.connect(self.clicked_FWT)
        self.pbn_IFWT.clicked.connect(self.clicked_IFWT)
        self.pbn_pyramid.clicked.connect(self.click_pyramid)
        
        self.pbn_FWT.setEnabled(False)
        self.pbn_IFWT.setEnabled(False)
        self.pbn_pyramid.setEnabled(False)

    def clicked_FWT(self):
        assert not self.srcImg is None 
        self.level = int(self.lineEdit_level.text())
        self.wvts = FWT2D(self.grayImg, level=self.level)
        wvtsImg = gen_waveletsImg(self.wvts, self.grayImg.shape)
        cv2.imshow('haar wavelets',wvtsImg)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def clicked_IFWT(self):
        assert not self.wvts is None
        img_r = IFWT2D(self.wvts)
        img_show(self.label_outputImg, img_r)


    def click_pyramid(self):
        assert not self.srcImg is None
        level = int(self.lineEdit_level_pyramid.text())
        kernel_size = int(self.lineEdit_kernel_size.text())
        kernel = gen_gauss_kernel(kernel_size=kernel_size)
        gauss_list, laplace_list = gen_pyramid(self.grayImg.copy(), level, kernel=kernel)
        pyramidImg = get_pyramidImg(gauss_list, laplace_list)
        cv2.imshow('pyramid image', pyramidImg)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def load_img(self):
        self.srcImgPath, self.srcImgType = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd()+'/src/',
            "Image Files(*.jpg *.png *.tif)")

        self.srcImg = cv2.imread(self.srcImgPath)
        self.srcImg = cv2.cvtColor(self.srcImg, cv2.COLOR_BGR2RGB)
        self.srcImg_float = self.srcImg.copy().astype(np.float)/255.0
        self.grayImg = self.srcImg_float[:,:,0]*0.30 + self.srcImg_float[:,:,1]*0.59 + self.srcImg_float[:,:,2]*0.11
        img_show(self.label_inputImg, self.srcImg)
        if not self.srcImg is None:
            self.pbn_FWT.setEnabled(True)
            self.pbn_IFWT.setEnabled(True)
            self.pbn_pyramid.setEnabled(True)            