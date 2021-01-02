import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPalette, QBrush, QPixmap, QImage
from PyQt5 import QtCore, QtGui, QtWidgets
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
pi = np.pi


# ============================================================================================
#                                           FFT
# ============================================================================================
def DFT(sig):
    N = sig.size
    V = np.array([[np.exp(-2j*pi*v*y/N) for v in range(N)] for y in range(N)])
    return sig.dot(V)

def FFT_r(img):
    if img.shape[1] == 1: 
        return img
    Wo, W, x = img.shape[1], img.shape[1], img
    if W & (W-1) : # not a power of 2, expand with zero
        W = 2**(int(np.log2(Wo))+1)
        x = np.zeros((img.shape[0], W), np.complex)
        x[:,:Wo] = img
    F_even = FFT_r(x[:,::2])
    F_odd =  FFT_r(x[:,1::2])
    D = np.diag(np.exp(-2j * pi * np.arange(int(W/2)) / W))
    return np.hstack([F_even + F_odd.dot(D),  F_even - F_odd.dot(D)])

def FFT2D(img):
    return FFT_r(FFT_r(img).T).T

def FFT_SHIFT(img):
    W,H = int(img.shape[0]/2), int(img.shape[1]/2)
    return np.vstack((np.hstack((img[W:,H:],img[W:,:H])),np.hstack((img[:W,H:],img[:W,:H]))))

def IFFT_r(img):
    if img.shape[1] == 1: 
        return img
    Wo, W, x = img.shape[1], img.shape[1], img
    if W & (W-1) : # not a power of 2, expand
        W = 2**(int(np.log2(Wo))+1)
        x = np.zeros((img.shape[0], W), np.complex)
        x[:,:Wo] = img

    F_even = IFFT_r(x[:,::2])
    F_odd =  IFFT_r(x[:,1::2])
    D = np.diag(np.exp(2j * pi * np.arange(int(W/2)) / W))
    return np.hstack([F_even + F_odd.dot(D),  F_even - F_odd.dot(D)])

def IFFT2D(F,shape=None):
    if shape is None:
        shape = F.shape
    img_o =  IFFT_r(IFFT_r(F).T).T
    img_o = img_o[0:shape[0], 0:shape[1]]
    real = np.abs(img_o.real)
    real = (real-real.min()) / (real.max() - real.min())
    real = (real*255).astype(np.uint8)        
    return real


# ============================================================================================
#                                           spatial
# ============================================================================================

def gen_gauss_kernel(kernel_size=3, sigma=1, k=1):
    if sigma == 0:
        sigma = 1
    X = np.linspace(-k, k, kernel_size)
    Y = np.linspace(-k, k, kernel_size)
    x, y = np.meshgrid(X, Y)
    kernel = np.exp(- (x**2 + y**2)/(2*sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel

def gauss_filter(img, kernel):
    kernel_size = kernel.shape[0]
    pad_size = int(kernel_size/2)
    H, W = img.shape
    tmpImg = np.zeros((H+2*pad_size,W+2*pad_size), dtype=np.float)
    tmpImg[pad_size:-pad_size, pad_size:-pad_size] = img 
    dstImg = img.copy()*kernel[pad_size, pad_size]
    for i in range(-pad_size, pad_size+1):
        for j in range(-pad_size, pad_size+1):
            dstImg += kernel[pad_size+i, pad_size+j]*tmpImg[pad_size+i : H+i+pad_size,    pad_size+j : W+j+pad_size]
    return dstImg


def downpool(img, step=2):
    H, W = img.shape
    dst = img[list(range(0,H,step)),:]
    dst = dst[:, list(range(0,W,step))]
    return dst

def uppool(img,step=2):
    H, W = img.shape
    dst = np.zeros((H*step, W*step))
    index = np.zeros_like(dst, dtype=np.bool)
    index1, index2 = index.copy(), index.copy()
    index1[list(range(0, H*step, step)), :] = True
    index2[:, list(range(0, W*step, step))] = True
    index = index1 & index2
    dst[index] = img.reshape(1,-1).squeeze()
    return dst


def gen_pyramid(img, level, kernel):
    gauss = img.copy()
    gauss_list, laplace_list = [], []
    for i in range(level):
        H,W = gauss.shape
        gauss = gauss_filter(gauss, kernel)
        if H%2==1:
            gauss = np.vstack((gauss, gauss[-1,:].reshape(1,-1)))
            H+=1
        if W%2 ==1:
            gauss = np.hstack((gauss, gauss[:,-1].reshape(-1,1)))
            W+=1
        gauss_dp2 = downpool(gauss,step=2)
        up2  = gauss_filter(uppool(gauss_dp2), kernel*4)
        laplace = gauss - up2
        gauss_list.append(gauss)
        laplace_list.append(laplace)
        gauss = gauss_dp2
    laplace_list[-1] = gauss_list[-1]
    return gauss_list, laplace_list



# ============================================================================================
#                                           wavelets
# ============================================================================================

def FWT1D(data, level=2):
    assert level <= np.log2(len(data))
    a = np.array(data)
    rst = []
    for i in range(level):
        if len(a)%2 == 1:
            a = np.append(a, a[-1])
        tmp = np.zeros_like(a)
        tmp[:len(a)-1] = a[1:]
        d = ((a - tmp)/np.sqrt(2))[range(0,len(a),2)]
        a = ((a + tmp)/np.sqrt(2))[range(0,len(a),2)]
        rst.append(d)
    rst.append(a)
    return list(reversed(rst)), level

def FWT2D_col(img):
    H, W = img.shape
    if W%2 == 1:
        img = np.hstack((img, img[:, -1].reshape(-1,1) ))
        W += 1
    tmp = np.zeros_like(img)
    tmp[:,:-1] = img[:,1:]
    vd = ((img - tmp)/np.sqrt(2))[:,range(0,W,2)]
    ah = ((img + tmp)/np.sqrt(2))[:,range(0,W,2)]
    return ah, vd

def FWT2D(img, level=2):
    H, W  = img.shape
    assert level <= min(np.log2(H), np.log2(W))
    a = img
    rst = []
    for i in range(level):
        ah, vd = FWT2D_col(a)
        a , h  = FWT2D_col(ah.T)
        v , d  = FWT2D_col(vd.T)
        a, h, v, d = a.T, h.T, v.T, d.T
        rst.append([h,v,d])
    rst.append(a)
    return list(reversed(rst))


def IFWT2D_row(a, d):
    assert a.shape == d.shape
    H, W = a.shape
    ae, de = np.zeros((H*2, W)), np.zeros((H*2, W))
    ae[range(0,2*H,2),:] = a; ae[range(1,2*H,2),:] = a
    de[range(0,2*H,2),:] = d; de[range(1,2*H,2),:] = -d
    return (ae+de)/np.sqrt(2)

def IFWT2D(wvts):
    lv = len(wvts)
    a = wvts[0]
    for i in range(1, lv):
        h, v, d = wvts[i]
        H, W = h.shape
        a = a[:H,:W]
        vd = IFWT2D_row(v, d)
        ah = IFWT2D_row(a, h)
        a = IFWT2D_row(ah.T, vd.T).T
    a = float2uint8(a)
    # a = (a*255).astype(np.uint8)
    return a



# ============================================================================================
#                                           UI tools
# ============================================================================================

def float2uint8(img):
    img[img<0] = 0
    img[img>1] = 1
    img = (img*255).astype(np.uint8)
    return img

def img_show(label, img):
    ## read W H of label
    label_width = label.width()
    label_height = label.height()
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    temp_img = QImage(img, img.shape[1], img.shape[0], img.shape[1]*3, QImage.Format_RGB888)
    pixmap_img = QPixmap.fromImage(temp_img).scaled(label_width, label_height)
    label.setPixmap(pixmap_img)

def gen_waveletsImg(wvts, shape):
    rst =  copy.deepcopy(wvts)
    img_rst = np.zeros(shape)
    level = len(rst)
    H, W = shape

    for lv in reversed(range(1,level)):
        Ht, Wt = int(H/2**(level-lv-1)), int(W/2**(level-lv-1))
        Mh, Mw = int(Ht/2), int(Wt/2)
        Hi, Wi = min(rst[lv][0].shape[0], Mh), min(rst[lv][0].shape[1], Mw)
        for i in range(3):
            rst[lv][i] = ((rst[lv][i]- rst[lv][i].min())/(rst[lv][i].max() - rst[lv][i].min()))**0.5

        img_rst[0 :0 +Hi, Mw:Mw+Wi] = rst[lv][0][:Hi,:Wi]   
        img_rst[Mh:Mh+Hi, 0 :0 +Wi] = rst[lv][1][:Hi,:Wi] 
        img_rst[Mh:Mh+Hi, Mw:Mw+Wi] = rst[lv][2][:Hi,:Wi] 
        img_rst[0 :0 +Ht, Mw-1:Mw+1] = 1
        img_rst[Mh-1:Mh+1, 0 :0 +Wt] = 1
    Hi, Wi = min(rst[0].shape[0], int(H/2**(level-1))), min(rst[0].shape[1], int(W/2**(level-1)))
    img_rst[:Hi, :Wi] = rst[0][:Hi,:Wi]/rst[0].max()
    return img_rst    

def get_pyramidImg(gauss_list, laplace_list):
    lv = len(gauss_list)
    H,W = gauss_list[0].shape
    rstImg = np.zeros((2*H, 2*W))
    h , w = H, 0
    for i in range(lv):
        gauss = gauss_list[i]
        laplace = laplace_list[i]
        laplace = (laplace - laplace.min())/(laplace.max()-laplace.min())
        laplace = laplace**0.3
        H, W = gauss.shape
        rstImg[h-H:h, w:w+W] = gauss
        rstImg[h:h+H, w:w+W] = laplace
        w += W
    return rstImg