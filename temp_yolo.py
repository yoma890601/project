import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
import numpy as np
import cv2
from qreader import QReader
cap = cv2.VideoCapture(2)
qrcode = cv2.QRCodeDetector()                        # 建立 QRCode 偵測器
qreader_reader = QReader()

def boxSize(arr):
   global data
   box_roll = np.rollaxis(arr,1,0)   # 轉置矩陣，把 x 放在同一欄，y 放在同一欄
   xmax = int(np.amax(box_roll[0]))  # 取出 x 最大值
   xmin = int(np.amin(box_roll[0]))  # 取出 x 最小值
   ymax = int(np.amax(box_roll[1]))  # 取出 y 最大值
   ymin = int(np.amin(box_roll[1]))  # 取出 y 最小值
   return (xmin,ymin,xmax,ymax)

# 如果 bbox 是 None 表示圖片中沒有 QRCode

# 從攝影機擷取一張影像
while(True):

   ret,im0 = cap.read()
   decoded_text = qreader_reader.detect(image=im0)
   # print(decoded_text)
   if decoded_text != []:
      # print(decoded_text)
      print(decoded_text[0]['bbox_xyxy'])
   # data, bbox, rectified = qrcode.detectAndDecode(im0)  # 偵測圖片中的 QRCode
   # if bbox is not None:
   #    print(data)                # QRCode 的內容
   #    # print(bbox)
   #    box = boxSize(bbox[0])
   #    cv2.rectangle(im0,(box[0],box[1]),(box[2],box[3]),(0,0,255),5)  # 畫矩形
   cv2.imshow('frame', im0)
   if cv2.waitKey(1) & 0xFF == ord('q'):
      break