import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPalette, QBrush, QPixmap, QImage
from PyQt5 import QtCore, QtGui, QtWidgets
import os
from ui_restoration import Ui_Restoration
import cv2
import numpy as np
from utils import *
pi = np.pi

class UiRestoration(QtWidgets.QWidget, Ui_Restoration):
    def __init__(self):
        super(UiRestoration,self).__init__()
        self.setupUi(self)
        self.srcImg = None
        self.srcImg_float = None
        self.grayImg = None
        self.tmpImg = None
        self.dstImg = None

        self.pbn_loadImg.clicked.connect(self.load_img)
        self.pbn_addNoise.clicked.connect(self.addNoise)
        self.pbn_average.clicked.connect(self.averageFilter)
        self.pbn_contraharmonic.clicked.connect(self.contraharmonicFilter)
        self.pbn_midPoint.clicked.connect(self.midPointFilter)
        self.pbn_AdaptiveMedian.clicked.connect(self.AdaptiveMedian)
        self.pbn_dst2src.clicked.connect(self.dstImg2srcImg)

        self.pbn_FFT.clicked.connect(self.click_FFT2D)
        self.pbn_idealBandReject.clicked.connect(self.ideal_bandReject)
        self.pbn_idealNotch.clicked.connect(self.ideal_notch)
        self.pbn_gaussBandReject.clicked.connect(self.gauss_bandReject)
        self.pbn_gaussNotch.clicked.connect(self.guass_notch)
        self.pbn_BbandReject.clicked.connect(self.butterworth_bandReject)
        self.pbn_BNotch.clicked.connect(self.butterworth_notch)

        self.pbn_AdaptiveMedian.setToolTip("很慢，请耐心等待")
        self.pbn_addNoise.setToolTip("右侧选择噪声，在原图增加噪声，并显示直方图")
        self.lineEdit_u0.setToolTip("点或区间，如30、60、20-100、50-(到边界)")
        self.lineEdit_v0.setToolTip("点或区间，如30、60、20-100、50-(到边界)")
        
        self.pbn_addNoise.setEnabled(False)
        self.pbn_average.setEnabled(False)
        self.pbn_contraharmonic.setEnabled(False)
        self.pbn_midPoint.setEnabled(False)     
        self.pbn_AdaptiveMedian.setEnabled(False)
        self.pbn_dst2src.setEnabled(False)

        self.pbn_FFT.setEnabled(False)     
        self.pbn_idealBandReject.setEnabled(False)          
        self.pbn_idealNotch.setEnabled(False)
        self.pbn_gaussBandReject.setEnabled(False)
        self.pbn_gaussNotch.setEnabled(False)
        self.pbn_BbandReject.setEnabled(False)        
        self.pbn_BNotch.setEnabled(False)
 


    def addNoise(self):
        # ['椒盐' '均匀' '高斯' '瑞利' '指数']
        index = self.comboBox_noise.currentIndex()
        H, W = self.grayImg.shape
        gray = self.grayImg
        uniform1 = np.random.rand(H,W)
        if index == 0: # 椒盐
            uniform2 = np.random.rand(H,W)
            p = 0.2
            noise = (uniform1<p)*(uniform1*0.2+0.8) + (uniform2<p)*(-uniform2*0.2-0.8)
        elif index == 1: # 均匀
            noise = uniform1
        elif index == 2: # 瑞利
            rayleigh = np.sqrt(-2*np.log(1-uniform1))
            noise = rayleigh/rayleigh.max()
        elif index == 3: # 高斯
            uniform2 = np.random.rand(H,W)
            rayleigh = np.sqrt(-2*np.log(1-uniform1))
            gauss = np.sin(2*pi*uniform2)*rayleigh
            noise = gauss/gauss.max()
        elif index == 4: #指数
            expo = -np.log(1-uniform1)
            noise = expo / expo.max()
        else:
            Exception('something wrong!')

        gray = gray +  noise*0.3
        hist, edges = self.histogram(gray,bins=100)
        histImg = self.getPltImg(hist)
        gray[gray<0]=0
        gray[gray>1]=1
        gray = (gray*255).astype(np.uint8)
        self.dstImg = gray.copy()
        img_show(self.label_tmpImg, histImg)
        img_show(self.label_outputImg, gray)


    def histogram(self, data, bins=10):
        edges = np.linspace(data.min(), data.max(), num=bins+1)
        hist = [np.sum((data>=edges[i]) & (data<=edges[i+1])) for i in range(len(edges)-1)]
        return hist, edges
    

    def getPltImg(self,data):
        fig,ax = plt.subplots(1)
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
        ax.axis('off')
        ax.plot(data)
        fig.canvas.draw()
        size_inches  = fig.get_size_inches()
        dpi          = fig.get_dpi()
        width, height = fig.get_size_inches() * fig.get_dpi()

        mplimage = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        imarray = np.reshape(mplimage, (int(height), int(width), 3))
        plt.close(fig)     
        return imarray     

    def averageFilter(self):
        window_size = int(self.lineEdit_S.text())
        assert window_size%2 == 1
        pad_size = int(window_size/2)
        H, W, C = self.srcImg.shape
        tmpImg = np.zeros((H+2*pad_size,W+2*pad_size,C), dtype=np.float)
        tmpImg[pad_size:-pad_size, pad_size:-pad_size, :] = self.srcImg_float 
        self.dstImg = self.srcImg_float.copy()
        for i in range(-pad_size, pad_size+1):
            for j in range(-pad_size, pad_size+1):
                self.dstImg += tmpImg[pad_size+i : H+i+pad_size,    pad_size+j : W+j+pad_size]
        self.dstImg = float2uint8(self.dstImg/window_size**2)
        img_show(self.label_outputImg, self.dstImg)        

    def contraharmonicFilter(self):
        window_size = int(self.lineEdit_S.text())
        assert window_size%2 == 1
        Q = int(self.lineEdit_Q.text())
        pad_size = int(window_size/2)
        H, W, C = self.srcImg.shape
        tmpImg = np.zeros((H+2*pad_size,W+2*pad_size,C), dtype=np.float)
        tmpImg[pad_size:-pad_size, pad_size:-pad_size, :] = self.srcImg_float
        stackedImg =  np.zeros((H,W,C,window_size**2))
        self.dstImg = self.srcImg_float.copy()
        for i in range(-pad_size, pad_size+1):
            for j in range(-pad_size, pad_size+1):
                stackedImg[:,:,:, (i+pad_size)*window_size+(j+pad_size)] = tmpImg[pad_size+i : H+i+pad_size,    pad_size+j : W+j+pad_size]
        self.dstImg = np.sum(stackedImg**(Q+1), axis=3) / (np.sum(stackedImg**(Q), axis=3) + 1e-5)
        self.dstImg = float2uint8(self.dstImg)
        img_show(self.label_outputImg, self.dstImg)

    def midPointFilter(self):
        window_size = int(self.lineEdit_S.text())
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
            self.dstImg[:,:,c] = 0.5*(np.max(stackedImg[:,:,c,:],axis=2) + np.min(stackedImg[:,:,c,:], axis=2))
        self.dstImg = float2uint8(self.dstImg)
        img_show(self.label_outputImg, self.dstImg)        


    def AdaptiveMedian(self):
        initWindowSize = 3
        maxWindowSize = 9
        pad_size = int(maxWindowSize/2)
        H, W, C = self.srcImg.shape
        self.dstImg = self.srcImg.copy()
        tmpImg = np.zeros((H+2*pad_size,W+2*pad_size,C), dtype=np.uint8)
        tmpImg[pad_size:-pad_size,pad_size:-pad_size,:] = self.srcImg.copy()
        for c in range(C):
            print('c', c)
            for h in range(H):
                for w in range(W):
                    ws = initWindowSize
                    while True:
                        ps = int(ws/2)
                        wv = tmpImg[h+pad_size-ps:h+pad_size+ps+1 , w+pad_size-ps:w+pad_size+ps+1, c]
                        z_max, z_min, z_mid = np.max(wv), np.min(wv), np.median(wv)
                        if ws >= maxWindowSize:
                            self.dstImg[h,w,c] = z_mid
                            break
                        if (z_min < z_mid) and (z_mid < z_max):
                            if (not ((z_min < tmpImg[h+pad_size, w+pad_size,c]) and (tmpImg[h+pad_size, w+pad_size,c]<z_max)) ):
                                self.dstImg[h,w,c] = z_mid
                                break
                        ws += 2
        img_show(self.label_outputImg, self.dstImg)


    def ideal_bandReject(self):
        self.click_FFT2D(show=False)
        D0 = int(self.lineEdit_D0.text())
        Window = int(self.lineEdit_W.text())
        H, W = self.F_shift.shape
        u, v = np.arange(H), np.arange(W)
        v, u = np.meshgrid(v, u)
        index =  ( np.sqrt((u-H/2)**2 + (v-W/2)**2) >= D0 - Window/2) & ( np.sqrt((u-H/2)**2 + (v-W/2)**2) <= D0 + Window/2)
        Hfilter = np.ones_like(self.F_shift)
        Hfilter[index] = 0
        self.frequency_filter(Hfilter)        

    def gauss_bandReject(self):
        self.click_FFT2D(show=False)
        D0 = int(self.lineEdit_D0.text())
        Window = int(self.lineEdit_W.text())
        H, W = self.F_shift.shape
        u, v = np.arange(H), np.arange(W)
        v, u = np.meshgrid(v, u)
        D = (u-H/2)**2 + (v-W/2)**2
        Hfilter = 1 -  np.exp(-0.5*((D-D0**2)**2)/(D*Window**2 + 1e-5))
        self.frequency_filter(Hfilter)

    def butterworth_bandReject(self):
        self.click_FFT2D(show=False)
        D0 = int(self.lineEdit_D0.text())
        Window = int(self.lineEdit_W.text())
        n = int(self.lineEdit_n.text())
        H, W = self.F_shift.shape
        u, v = np.arange(H), np.arange(W)
        v, u = np.meshgrid(v, u)
        D = (u-H/2)**2 + (v-W/2)**2
        Hfilter =  1/((D*Window**2/((D-D0**2)**2+ 1e-5))**n + 1)
        self.frequency_filter(Hfilter) 
    
    
    def getNotchParams(self,shape):
        D0 = int(self.lineEdit_D0.text())
        H, W = shape
        u0_list = list(self.lineEdit_u0.text().split('-'))
        v0_list = list(self.lineEdit_v0.text().split('-'))
        u0_list[0] = int(u0_list[0])
        if (len(u0_list) == 1):
            u0_list.append(u0_list[0]+1)
        if (len(u0_list) == 2) and (u0_list[1]==''):
            u0_list[1] = int(H/2)
        
        v0_list[0] = int(v0_list[0])
        if(len(v0_list)==1):
            v0_list.append(v0_list[0]+1)
        if (len(v0_list) == 2) and (v0_list[1]==''):
            v0_list[1] = int(W/2)
        return D0, u0_list, v0_list


    def ideal_notch(self):
        self.click_FFT2D(show=False)
        H, W = self.F_shift.shape
        D0, u0_list, v0_list = self.getNotchParams(shape=(H,W))
        u, v = np.arange(H), np.arange(W)
        v, u = np.meshgrid(v, u)
        Hfilter = np.ones_like(self.F_shift)
        Index = np.zeros_like(self.F_shift,dtype=np.bool)
        for u0 in range(u0_list[0], u0_list[1], D0):
            for v0 in range(v0_list[0], v0_list[1], D0):
                index =  ((u-H/2-u0)**2 + (v-W/2-v0)**2 <=D0**2 ) | ( (u-H/2+u0)**2 + (v-W/2+v0)**2 <=D0**2  )
                Index = Index | index
        Hfilter[Index] = 0
        self.frequency_filter(Hfilter)     

    def guass_notch(self):
        self.click_FFT2D(show=False)
        H, W = self.F_shift.shape
        D0, u0_list, v0_list = self.getNotchParams(shape=(H,W))      
        u, v = np.arange(H), np.arange(W)
        v, u = np.meshgrid(v, u)
        Hfilter = np.ones_like(self.F_shift)
        for u0 in range(u0_list[0], u0_list[1], D0):
            for v0 in range(v0_list[0], v0_list[1], D0):        
                D1 = np.sqrt((u-H/2-u0)**2 + (v-W/2-v0)**2)
                D2 = np.sqrt((u-H/2+u0)**2 + (v-W/2+v0)**2)
                hfilter = (1 -  np.exp(-0.5*(D1*D2/D0**2)))
                Hfilter = Hfilter * hfilter
        self.frequency_filter(Hfilter)

    def butterworth_notch(self):
        self.click_FFT2D(show=False)        
        n = int(self.lineEdit_n.text())
        H, W = self.F_shift.shape
        D0, u0_list, v0_list = self.getNotchParams(shape=(H,W))
        u, v = np.arange(H), np.arange(W)
        v, u = np.meshgrid(v, u)
        Hfilter = np.ones_like(self.F_shift)
        for u0 in range(u0_list[0], u0_list[1], D0):
            for v0 in range(v0_list[0], v0_list[1], D0):        
                D1 = np.sqrt((u-H/2-u0)**2 + (v-W/2-v0)**2)
                D2 = np.sqrt((u-H/2+u0)**2 + (v-W/2+v0)**2)
                hfilter =  1/( 1+ (D0**2/(D1*D2+1e-5))**n )
                Hfilter = Hfilter * hfilter
        self.frequency_filter(Hfilter)         

    def frequency_filter(self, Hfilter):
        F_shift = self.F_shift.copy()
        F_shift *= Hfilter
        F = FFT_SHIFT(F_shift)
        img_lp = IFFT2D(F,shape=self.srcImg.shape)
        self.dstImg = img_lp
        if img_lp.shape == 2:
            self.dstImg = cv2.cvtColor(self.dstImg, cv2.COLOR_GRAY2RGB)
        self.show_Fimg(F_shift, self.label_tmpImg)
        img_show(self.label_outputImg, img_lp)     

    def show_Fimg(self, F, label):
        Fm = np.sqrt(F.real**2 + F.imag**2)
        Fr = np.log(Fm + 1)
        Fimg = (Fr/Fr.max()*255).astype(np.uint8)
        img_show(label, Fimg)

    def click_FFT2D(self, show=True):
        assert not self.srcImg is None
        self.F = FFT2D(self.grayImg)
        self.F_shift = FFT_SHIFT(self.F)
        # if show:
        self.show_Fimg(self.F_shift, self.label_tmpImg)
    

    def click_IFFT2D(self, show=True):
        assert not self.F is None
        img = IFFT2D(self.F, shape=self.srcImg.shape) 
        return img

    def dstImg2srcImg(self):
        assert not self.dstImg is None
        if len(self.dstImg.shape) == 2:
            self.grayImg = self.dstImg.copy().astype(np.float)/255
            self.srcImg = cv2.cvtColor(cv2.cvtColor(self.dstImg, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
            self.srcImg_float = self.srcImg.copy().astype(np.float)/255.0
        elif len(self.dstImg.shape) == 3:
            self.srcImg = self.dstImg.copy()
            self.srcImg_float = self.dstImg.copy().astype(np.float)/255.0
            self.grayImg = self.srcImg_float[:,:,0]*0.30 + self.srcImg_float[:,:,1]*0.59 + self.srcImg_float[:,:,2]*0.11
        img_show(self.label_inputImg, self.srcImg)


    def load_img(self):
        self.srcImgPath, self.srcImgType = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd()+'/src/',
            "Image Files(*.jpg *.png *.tif)")

        self.srcImg = cv2.imread(self.srcImgPath)
        self.srcImg = cv2.cvtColor(self.srcImg, cv2.COLOR_BGR2RGB)
        self.srcImg_float = self.srcImg.copy().astype(np.float)/255.0
        self.grayImg = self.srcImg_float[:,:,0]*0.30 + self.srcImg_float[:,:,1]*0.59 + self.srcImg_float[:,:,2]*0.11
        img_show(self.label_inputImg, self.srcImg)
        if not self.srcImg is None:
            self.pbn_addNoise.setEnabled(True)
            self.pbn_average.setEnabled(True)
            self.pbn_contraharmonic.setEnabled(True)
            self.pbn_midPoint.setEnabled(True)     
            self.pbn_AdaptiveMedian.setEnabled(True)
            self.pbn_dst2src.setEnabled(True)

            self.pbn_FFT.setEnabled(True)     
            self.pbn_idealBandReject.setEnabled(True)          
            self.pbn_idealNotch.setEnabled(True)
            self.pbn_gaussBandReject.setEnabled(True)
            self.pbn_gaussNotch.setEnabled(True)
            self.pbn_BbandReject.setEnabled(True)        
            self.pbn_BNotch.setEnabled(True)