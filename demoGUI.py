#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import cv2
from PyQt5 import QtCore, QtGui,QtWidgets
import os
import numpy as np
reload(sys)
sys.setdefaultencoding('utf-8')

from CTPN.ctpninference import *
from crnn.crnninference import *

#default target size
WIDTH = 640
HEIGHT = 480
SAVE_TXT = 'reco.txt'

#initialize the network should set the target_size（net input size）
#ctpn
text_detector = ctpnModel()
#crnn
model,converter = crnnModel()

class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        self.timer_camera = QtCore.QTimer()
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.set_ui()
        self.slot_init()
        self.__flag_work = 0
        self.x =0

    def set_ui(self):
        self.__layout_main = QtWidgets.QHBoxLayout()
        self.__layout_fun_button = QtWidgets.QVBoxLayout()
        self.__layout_data_show = QtWidgets.QVBoxLayout()
        self.button_open_camera = QtWidgets.QPushButton(u'打开相机')
        self.button_open_video = QtWidgets.QPushButton(u'打开图片或视频')
        self.button_close_any = QtWidgets.QPushButton(u'关闭')
        self.button_close = QtWidgets.QPushButton(u'退出')
        self.button_open_video.setMinimumHeight(50)
        self.button_close_any.setMinimumHeight(50)
        self.button_open_camera.setMinimumHeight(50)
        self.button_close.setMinimumHeight(50)
        self.button_close.move(10,100)

        # 信息显示
        self.label_show_camera = QtWidgets.QLabel()
        self.label_move = QtWidgets.QLabel()
        self.label_move.setFixedSize(200, 200)

        self.label_show_camera.setFixedSize(961, 721)
        self.label_show_camera.setAutoFillBackground(False)

        self.__layout_fun_button.addWidget(self.button_open_camera)
        self.__layout_fun_button.addWidget(self.button_open_video)
        self.__layout_fun_button.addWidget(self.button_close_any)
        self.button_close_any.setVisible(False)
        self.__layout_fun_button.addWidget(self.button_close)
        self.__layout_fun_button.addWidget(self.label_move)

        self.__layout_main.addLayout(self.__layout_fun_button)
        self.__layout_main.addWidget(self.label_show_camera)

        self.setLayout(self.__layout_main)
        self.label_move.raise_()
        self.setWindowTitle(u'场景文字识别')

    def slot_init(self):
        self.button_open_camera.clicked.connect(self.button_open_camera_click)
        self.timer_camera.timeout.connect(self.show_camera)
        self.button_open_video.clicked.connect(self.button_open_video_click)
        self.button_close_any.clicked.connect(self.closeVideo)
        self.button_close.clicked.connect(self.close)

    def closeVideo(self):
        if self.cap.isOpened():
            self.cap.release()
        if self.timer_camera.isActive():
            self.timer_camera.stop()
        self.button_open_video.setVisible(True)
        self.button_open_camera.setVisible(True)
        self.button_close_any.setVisible(False)

    def save_txt(self,contents):
        with open(SAVE_TXT, 'a') as f:
            f.write('\n')
            f.write(contents)
            f.close()

    def toSize(self,img):
	print(img.shape)
	if(img.shape[1]>960):
	    scale = 960.0/img.shape[1]
	    height = int(img.shape[0]*scale)
	    return height, 960
	else:
	    return img.shape[0], img.shape[1] 

    def button_open_video_click(self):
        if self.timer_camera.isActive() == False:
            fileName, filetype = QtWidgets.QFileDialog.getOpenFileName(self,
                "选取文件","C:/","All Files (*);;Text Files (*.mp4)")
            print(fileName, filetype)
            flag = self.cap.open(fileName)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检查图片或视频格式是否正确", buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(30)
                self.button_open_video.setVisible(False)
                self.button_open_camera.setVisible(False)
                self.button_close_any.setVisible(True)
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.label_show_camera.clear()
            self.button_open_video.setText(u'打开图片或视频')

    def button_open_camera_click(self):
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(30)
                self.button_open_video.setVisible(False)
                self.button_open_camera.setVisible(False)
                self.button_close_any.setVisible(True)
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.label_show_camera.clear()
            self.button_open_camera.setText(u'打开相机')

    # show image on mainwindow
    def show_camera(self):
        flag, self.image = self.cap.read()
        if flag:
            img,imsrc,text_recs = getTextRec(text_detector,self.image)
            _str = crnnRec(model,converter,img,text_recs)
            self.save_txt(_str)
	    show = self.image
	    newh, neww = self.toSize(self.image)
            show = cv2.resize(self.image, (neww, newh))
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
            self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))
        else:
            self.closeVideo()

    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cacel = QtWidgets.QPushButton()
        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"关闭", u"是否关闭！")
        msg.addButton(ok,QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'确定')
        cacel.setText(u'取消')
        # msg.setDetailedText('sdfsdff')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            #             self.socket_client.send_command(self.socket_client.current_user_command)
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())
