import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPalette, QBrush, QPixmap, QImage
from PyQt5 import QtCore, QtGui, QtWidgets
import os
from ui_frequency import Ui_Frequency
import cv2
import numpy as np
from utils import *
pi = np.pi


class UiFrequency(QtWidgets.QWidget, Ui_Frequency):
    def __init__(self):
        super(UiFrequency,self).__init__()
        self.setupUi(self)    
        self.srcImg = None
        self.srcImg_float = None
        self.grayImg = None
        self.F = None
        self.F_shift = None
        self.pbn_loadImg.clicked.connect(self.load_img)
        self.pbn_FFT.clicked.connect(self.click_FFT2D)
        self.pbn_IFFT.clicked.connect(self.click_IFFT2D)
        self.pbn_idealLP.clicked.connect(self.ideal_low_pass)
        self.pbn_BLP.clicked.connect(self.BLPF)
        self.pbn_gaussLP.clicked.connect(self.gauss_BP)
        self.pbn_idealHP.clicked.connect(self.ideal_high_pass)
        self.pbn_BHP.clicked.connect(self.BHPF)
        self.pbn_gaussHP.clicked.connect(self.gauss_HP)

        self.pbn_FFT.setEnabled(False)
        self.pbn_IFFT.setEnabled(False)
        self.pbn_idealLP.setEnabled(False)
        self.pbn_BLP.setEnabled(False)
        self.pbn_gaussLP.setEnabled(False)
        self.pbn_idealHP.setEnabled(False)
        self.pbn_BHP.setEnabled(False)
        self.pbn_gaussHP.setEnabled(False)

    def click_FFT2D(self):
        assert not self.srcImg is None
        self.F = FFT2D(self.grayImg)
        self.F_shift = FFT_SHIFT(self.F)
        if not self.F_shift is None:
            self.pbn_IFFT.setEnabled(True)
            self.pbn_idealLP.setEnabled(True)
            self.pbn_BLP.setEnabled(True)
            self.pbn_gaussLP.setEnabled(True)
            self.pbn_idealHP.setEnabled(True)
            self.pbn_BHP.setEnabled(True)
            self.pbn_gaussHP.setEnabled(True)            
        self.show_Fimg(self.F_shift)
    

    def click_IFFT2D(self):
        assert not self.F is None
        img = IFFT2D(self.F, shape=self.srcImg.shape)     
        img_show(self.label_outputImg, img)

    def filter(self, Hfilter):
        F_shift = self.F_shift.copy()
        F_shift *= Hfilter
        F = FFT_SHIFT(F_shift)
        img_lp = IFFT2D(F,shape=self.srcImg.shape)
        img_show(self.label_outputImg, img_lp)        

    def ideal_low_pass(self, HP=False):
        D0 = int(self.lineEdit_D0.text())
        H, W = self.F_shift.shape
        u, v = np.arange(H), np.arange(W)
        v, u = np.meshgrid(v, u)
        index = (u-H/2)**2 + (v-W/2)**2 < D0**2
        Hfilter = np.zeros_like(self.F_shift)
        Hfilter[index] = 1
        if HP:
            Hfilter = 1 - Hfilter
        self.filter(Hfilter)

    def BLPF(self, HP=False):
        n = int(self.lineEdit_n.text())
        D0 = int(self.lineEdit_D0.text())
        H, W = self.F_shift.shape
        u, v = np.arange(H), np.arange(W)
        v, u = np.meshgrid(v, u)        
        Hfilter = 1 / (1 + (((u-H/2)**2 + (v-W/2)**2)/D0**2)**n)
        if HP:
            Hfilter = 1 - Hfilter
        self.filter(Hfilter)                

    def gauss_BP(self, HP=False):
        n = 2
        D0 = int(self.lineEdit_D0.text())
        H, W = self.F_shift.shape
        u, v = np.arange(H), np.arange(W)
        v, u = np.meshgrid(v, u)        
        Hfilter = np.ones_like(self.F_shift)
        Hfilter *=  np.exp(-0.5*(((u-H/2)**2 + (v-W/2)**2)/D0**2))
        if HP:
            Hfilter = 1 - Hfilter
        self.filter(Hfilter)         


    def ideal_high_pass(self):
        self.ideal_low_pass(HP=True)

    def gauss_HP(self):
        self.gauss_BP(HP=True)

    def BHPF(self):
        self.BLPF(HP=True)

    def show_Fimg(self, F):
        Fm = np.sqrt(F.real**2 + F.imag**2)
        Fr = np.log(Fm + 1)
        Fimg = (Fr/Fr.max()*255).astype(np.uint8)
        img_show(self.label_outputImg, Fimg)

    def load_img(self):
        self.srcImgPath, self.srcImgType = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd()+'/src/',
            "Image Files(*.jpg *.png *.tif)")

        self.srcImg = cv2.imread(self.srcImgPath)
        self.srcImg = cv2.cvtColor(self.srcImg, cv2.COLOR_BGR2RGB)
        self.srcImg_float = self.srcImg.copy().astype(np.float)/255.0
        self.grayImg = self.srcImg_float[:,:,0]*0.30 + self.srcImg_float[:,:,1]*0.59+ self.srcImg_float[:,:,2]*0.11
        img_show(self.label_inputImg, self.srcImg)
        self.click_FFT2D()
        if not self.srcImg is None:
            self.pbn_FFT.setEnabled(True)            
