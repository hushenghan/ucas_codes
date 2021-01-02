import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPalette, QBrush, QPixmap, QImage
from PyQt5 import QtCore, QtGui, QtWidgets
import os
from ui_spatial import Ui_Spatial
import cv2
import numpy as np
from utils import *
pi = np.pi


class UiSpatial(QtWidgets.QWidget, Ui_Spatial):
    def __init__(self):
        super(UiSpatial,self).__init__()
        self.setupUi(self)
        self.srcImg = None
        self.srcImg_float = None

        self.pbn_loadImg.clicked.connect(self.load_img)
        self.pbn_gamma.clicked.connect(self.gamma_correction)
        self.pbn_lapacian.clicked.connect(self.lapacian)
        self.pbn_reverse.clicked.connect(self.light_reverse)
        self.pbn_hisEqu.clicked.connect(self.histogram_equalization)
        self.pbn_detailed.clicked.connect(self.detailed)
        self.pbn_gradientH.clicked.connect(self.gradient_H)
        self.pbn_gradientV.clicked.connect(self.gradient_V)
        self.pbn_gaussFilter.clicked.connect(self.gauss_filter)
        self.pbn_medianFilter.clicked.connect(self.median_filter)

        self.pbn_gamma.setEnabled(False)
        self.pbn_lapacian.setEnabled(False)
        self.pbn_reverse.setEnabled(False)
        self.pbn_hisEqu.setEnabled(False)
        self.pbn_detailed.setEnabled(False)
        self.pbn_gradientH.setEnabled(False)
        self.pbn_gradientV.setEnabled(False)
        self.pbn_gaussFilter.setEnabled(False)
        self.pbn_medianFilter.setEnabled(False)


    def gamma_correction(self):
        gamma = float(self.lineEdit_gamma.text())
        self.dstImg = self.srcImg_float.copy()
        self.dstImg = np.power(self.dstImg, gamma)
        self.dstImg = (self.dstImg*255).astype(np.uint8)
        img_show(self.label_outputImg, self.dstImg)


    def light_reverse(self):
        self.dstImg = 255 - self.srcImg.copy()
        img_show(self.label_outputImg, self.dstImg)


    def lapacian(self):
        self.dstImg = self.srcImg_float.copy()
        H, W, C = self.srcImg.shape
        tmpImg = np.zeros((H+2,W+2,C), dtype=np.float)
        tmpImg[1:-1,1:-1,:] = self.dstImg
        self.dstImg = -4*self.dstImg + \
                        tmpImg[1:-1, 0:-2] + tmpImg[1:-1,  2:] + \
                        tmpImg[0:-2, 1:-1]  + tmpImg[2:, 1:-1]
        self.dstImg = float2uint8(self.dstImg)
        img_show(self.label_outputImg, self.dstImg)
    
    def histogram_equalization(self):
        # gray = 0.30*R+0.59*G+0.11*B
        tmp = self.srcImg_float[:,:,0]*0.30 + self.srcImg_float[:,:,1]*0.59+ self.srcImg_float[:,:,2]*0.11
        tmp = (tmp*255).astype(np.uint8)
        CDFs = [np.sum(tmp<=i) for i in range(256)]
        mapped_gray = ((np.array(CDFs, dtype=np.float)/(tmp.shape[0]*tmp.shape[1]))*255).astype(np.uint8)
        self.dstImg = tmp.copy()
        for i in range(256):
            self.dstImg[tmp==i] = mapped_gray[i]
        img_show(self.label_outputImg, self.dstImg)


    def detailed(self):
        self.gauss_filter()
        self.dstImg = 2*(self.srcImg.astype(np.float)/255) - self.dstImg.astype(np.float)/255
        
        self.dstImg = float2uint8(self.dstImg)
        img_show(self.label_outputImg, self.dstImg)


    def gradient_H(self):
        self.dstImg = self.srcImg_float.copy()
        H, W, C = self.srcImg.shape
        tmpImg = np.zeros((H+2,W+2,C), dtype=np.float)
        tmpImg[1:-1,1:-1,:] = self.dstImg        
        self.dstImg = -tmpImg[0:-2, 0:-2]  + tmpImg[0:-2, 2:] \
                        -2*tmpImg[1:-1, 0:-2]  + 2*tmpImg[1:-1, 2:] \
                        -tmpImg[2:, 0:-2] + tmpImg[2:, 2:]
        self.dstImg = float2uint8(self.dstImg)
        img_show(self.label_outputImg, self.dstImg)

    def gradient_V(self):
        self.dstImg = self.srcImg_float.copy()
        H, W, C = self.srcImg.shape
        tmpImg = np.zeros((H+2,W+2,C), dtype=np.float)
        tmpImg[1:-1,1:-1,:] = self.dstImg
        self.dstImg = -tmpImg[0:-2, 0:-2] - 2*tmpImg[0:-2, 1:-1] - tmpImg[0:-2, 2:] \
                        +tmpImg[2:, 0:-2] + 2*tmpImg[2:, 1:-1] + tmpImg[2:, 2:]
        self.dstImg = float2uint8(self.dstImg)
        img_show(self.label_outputImg, self.dstImg)

    def gauss_filter(self):
        kernel_size = int(self.lineEdit_gauss.text())
        assert kernel_size%2 == 1
        k = int(self.lineEdit_gauss_k.text())
        pad_size = int(kernel_size/2)
        H, W, C = self.srcImg.shape
        tmpImg = np.zeros((H+2*pad_size,W+2*pad_size,C), dtype=np.float)
        tmpImg[pad_size:-pad_size, pad_size:-pad_size, :] = self.srcImg_float 
        kernel = gen_gauss_kernel(kernel_size, sigma=1, k=k)
        self.dstImg = self.srcImg_float.copy()*kernel[pad_size, pad_size]
        for i in range(-pad_size, pad_size+1):
            for j in range(-pad_size, pad_size+1):
                self.dstImg += kernel[pad_size+i, pad_size+j]*tmpImg[pad_size+i : H+i+pad_size,    pad_size+j : W+j+pad_size]
        self.dstImg = float2uint8(self.dstImg)
        img_show(self.label_outputImg, self.dstImg)


    # def median_filter(self):
    #     window_size = int(self.lineEdit_meidanFilter.text())
    #     r = int(window_size/2)
    #     H, W, C = self.srcImg.shape
    #     tmpImg = np.zeros((H+2*r,W+2*r,C), dtype=np.uint8)
    #     tmpImg[r:-r, r:-r, :] = self.srcImg
    #     self.dstImg = np.zeros_like(self.srcImg, dtype=np.uint8)
    #     for k in range(C):
    #         for i in range(r, r+H):
    #             for j in range(r, r+W):   
    #                 self.dstImg[i-r,j-r,k] = np.median(tmpImg[i-r:i+r+1,j-r:j+r+1,k])
    #     img_show(self.label_outputImg, self.dstImg)


    def median_filter(self):
        window_size = int(self.lineEdit_meidanFilter.text())
        assert window_size%2 == 1
        pad_size = int(window_size/2)
        H, W, C = self.srcImg.shape
        tmpImg = np.zeros((H+2*pad_size,W+2*pad_size,C), dtype=np.float)
        tmpImg[pad_size:-pad_size, pad_size:-pad_size, :] = self.srcImg_float
        stackedImg =  np.zeros((H,W,C,window_size**2))
        self.dstImg = self.srcImg_float.copy()
        for i in range(-pad_size, pad_size+1):
            for j in range(-pad_size, pad_size+1):
                stackedImg[:,:,:, (i+pad_size)*window_size+(j+pad_size)] = tmpImg[pad_size+i : H+i+pad_size,    pad_size+j : W+j+pad_size]
        for c in range(C):
            self.dstImg[:,:,c] = np.median(stackedImg[:,:,c,:],axis=2) 
        self.dstImg = float2uint8(self.dstImg)
        img_show(self.label_outputImg, self.dstImg)        

    def load_img(self):
        self.srcImgPath, self.srcImgType = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd()+'/src/',
            "Image Files(*.jpg *.png *.tif)")

        self.srcImg = cv2.imread(self.srcImgPath)
        self.srcImg = cv2.cvtColor(self.srcImg, cv2.COLOR_BGR2RGB)
        self.srcImg_float = self.srcImg.copy().astype(np.float)/255.0
        img_show(self.label_inputImg, self.srcImg)
        if not self.srcImg is None:
            self.pbn_gamma.setEnabled(True)
            self.pbn_lapacian.setEnabled(True)
            self.pbn_reverse.setEnabled(True)
            self.pbn_hisEqu.setEnabled(True)
            self.pbn_detailed.setEnabled(True)
            self.pbn_gradientH.setEnabled(True)
            self.pbn_gradientV.setEnabled(True)
            self.pbn_gaussFilter.setEnabled(True)
            self.pbn_medianFilter.setEnabled(True)
