import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPalette, QBrush, QPixmap, QImage
from PyQt5 import QtCore, QtGui, QtWidgets
import os
import ui_entry
from spatial_method import UiSpatial
from frequency_method import UiFrequency
from restoration_method import UiRestoration
from wavelets_method import UiWavelets
import cv2
import numpy as np


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    main_entry = ui_entry.Ui_entryWindow()
    ui_spatial = UiSpatial()
    ui_frequency = UiFrequency()
    ui_restoration = UiRestoration()
    ui_wavelets = UiWavelets()

    main_entry.setupUi(MainWindow)
    MainWindow.show()
    main_entry.pbn_spatial.clicked.connect(ui_spatial.show)
    main_entry.pbn_frequency.clicked.connect(ui_frequency.show)
    main_entry.pbn_restoration.clicked.connect(ui_restoration.show)
    main_entry.pbn_wavelets.clicked.connect(ui_wavelets.show)
    
    sys.exit(app.exec_())