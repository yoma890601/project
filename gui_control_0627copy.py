# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'connect_me.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow,QInputDialog,QWidget, QApplication, QPushButton, QMessageBox, QLabel, QCheckBox,QGraphicsPixmapItem, QGraphicsScene,QDialog, QFileDialog, QGridLayout
from PyQt5.QtGui import QImage, QPixmap
from UI import Ui_MainWindow

import glob,os,sys,cv2,argparse
import numpy as np

from detect import classify 
from utils.plots import plot_one_box

# 必須在yolo底下 要用到 utils.plots

def xyxy2xywh_transfer(temp_xy,imgyx,mode):
    if mode == 'xyxy2xywh':
        xywh = [((temp_xy[0]+temp_xy[2])/2)/imgyx[0],((temp_xy[1]+temp_xy[3])/2)/imgyx[1],abs((temp_xy[0]-temp_xy[2])) /imgyx[0],abs((temp_xy[1]-temp_xy[3])) /imgyx[1]]
        return xywh
    elif mode == 'xywh2xyxy':
        xyxy = [(temp_xy[0]-temp_xy[2]/2)*imgyx[0] ,(temp_xy[1]-temp_xy[3]/2)*imgyx[1] , (temp_xy[0]+temp_xy[2]/2)*imgyx[0] , (temp_xy[1]+temp_xy[3]/2)*imgyx[1]]
        return xyxy
def check_path():
    load_Eclatorq_path = './Eclatorq/images'
    save_path = './Eclatorq'
    save_sop_path = './Eclatorq/sop'
    save_txt_path = './Eclatorq/sop/labels'
    temp_yoma_path ='./yoma_data/temp'
    need_path = [save_path,load_Eclatorq_path ,save_sop_path,save_txt_path,temp_yoma_path]
    print(need_path)

    necessary_env = os.path.split(os.getcwd())[1]
    print(necessary_env)
    if necessary_env =='yolov7':
        for i in range(len(need_path)):
            if not os.path.exists(need_path[i]):
                print("mkdir : "+ str(need_path[i]))
                os.makedirs(need_path[i])
            else :
                print("exists : "+ str(need_path[i]))
    else:
        print("This Project Must Used By yolov7 ")
        # 因為會用到 yolov7裡面的東西
        # 

class MyMainForm(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainForm, self).__init__(parent)
        self.setupUi(self)

        self.open_dir_Button.clicked.connect(self.open_folder)
        self.save_button.clicked.connect(self.save)
        self.close_button.clicked.connect(self.close)
        
        self.next_button.clicked.connect(self.next)
        self.last_button.clicked.connect(self.last)
        # self.classify_button.clicked.connect(self.classify)
        # self.define_button.clicked.connect(self.define)
        self.return_button.clicked.connect(self.yoma_return)
        self.label_button.clicked.connect(self.newlabel)

        self.graphicsView.mousePressEvent = self.mousePress # set_clicked_position
        self.graphicsView.mouseReleaseEvent = self.mouseRelease
        self.mode = 'NULL'
        self.pic_num = 0
        self.frame = []
        self.folder_path = ''
        self.label_flag = False
        self.order = 0
        self.define_list = []

        self.actionDebugMode.triggered.connect(self.debug)
        self.debug_state = 0

        check_path()

    def debug(self,state):
        # state= 0
        self.debug_state = state
        if self.debug_state == 1 :
            myWin.resize(1280, 720)
        else :
            myWin.resize(800, 720)
    def classify (self):
        print(self.debug_state)
        if self.debug_state == 0 :
            parser = argparse.ArgumentParser()
            parser.add_argument('--weights', nargs='+', type=str, default='20230517best.pt', help='model.pt path(s)') # yolov7.pt
            parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder  inference/images  , 0 for webcam
            parser.add_argument('--img-size', type=int, default=160, help='inference size (pixels)')
            parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
            parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
            parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
            parser.add_argument('--view-img', action='store_true', help='display results')
            parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
            parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
            parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
            parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
            parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
            parser.add_argument('--augment', action='store_true', help='augmented inference')
            parser.add_argument('--update', action='store_true', help='update all models')
            parser.add_argument('--project', default='runs/detect', help='save results to project/name')
            parser.add_argument('--name', default='exp', help='save results to project/name')
            parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
            parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
            opt = parser.parse_args()
            print(opt)
            if self.folder_path == '':
                opt.source = 'inference/images' #
            else :
                opt.source = self.folder_path
            print(opt.source)
            self.classify_result_dir = test_classify.detect(opt) 
            self.show_text.setText('Classify Path : \n'+opt.source+'\n'+'Done\n'+"Save To : "+'\n'+str( self.classify_result_dir))
        elif self.debug_state == 1 :
            self.classify_result_dir = "./runs/detect/debug"
        self.showImage()
        
    def showImage(self,show_flag='default'):
        self.image_paths = getImagesInDir(self.folder_path)
        list_len = len(self.image_paths) 
        if list_len > 0 :
            if self.pic_num >= list_len :
                self.pic_num = self.pic_num - list_len
            elif self.pic_num <= -list_len :
                self.pic_num = self.pic_num + list_len
            #temp = 0+self.pic_num
            # self.pic_g.setTitle(str(self.image_paths[self.pic_num]))
            self.show_text.setText(str(self.image_paths[self.pic_num]))
            self.frame = QImage(self.image_paths[self.pic_num])
            w = self.frame.width()
            h = self.frame.height()
            while w > 640 or h > 480 :
                w = w*0.8
                h = h*0.8
            # print(w,h)
            self.frame = self.frame.scaled(w, h, QtCore.Qt.KeepAspectRatio)
            pix = QPixmap.fromImage( self.frame)
            item = QGraphicsPixmapItem(pix)
            scene = QGraphicsScene()
            scene.addItem(item)
            self.graphicsView.setGeometry(QtCore.QRect(350-w/2, 270-h/2, w+5, h+5))
            self.graphicsView.setScene(scene)

            img = self.qimg2cv(self.frame)
            self.img_x,  self.img_y , _= img.shape
            self.to_txt(self.image_paths[self.pic_num])
            self.show_text.setText(str(self.classify_result_dir))
            self.disp_text.setText('Mode ：'+str(self.mode)+'  File Name：'+str(self.img_name))

            if show_flag=='default' :
                f = open(self.open_txt+'.txt')
                for line in f.readlines():
                    line = list(map(float, line.split(' ')))
                    self.xyxy = xyxy2xywh_transfer([float(line[1]),float(line[2]),float(line[3]),float(line[4])],[self.img_y,self.img_x],'xywh2xyxy')
                    cv2.rectangle(img, [int(self.xyxy[0]),int(self.xyxy[1])], [int(self.xyxy[2]),int(self.xyxy[3])], (0, 0, 255), 1)
                f.close
            elif show_flag=='label' :
                for self.label_c,self.label_x,self.label_y,self.label_w,self.label_h in self.new_label:
                    self.xyxy = xyxy2xywh_transfer([self.label_x,self.label_y,self.label_w,self.label_h],[self.img_y,self.img_x],'xywh2xyxy')
                    cv2.rectangle(img, [int(self.xyxy[0]),int(self.xyxy[1])], [int(self.xyxy[2]),int(self.xyxy[3])], (0, 0, 255), 1)
            cv2.imwrite('./yoma_data/temp/temp.jpg',img)
            self.show_newImage('./yoma_data/temp/temp.jpg')


        else :
            QMessageBox.warning(self, "Warning", "Can't Find The File")
            self.show_text.setText(self.folder_path+'\n'+" Can't Find The File ")
    def show_newImage(self,img):
        self.frame = QImage(img)
        w = self.frame.width()
        h = self.frame.height()

        self.frame = self.frame.scaled(w, h, QtCore.Qt.KeepAspectRatio)
        pix = QPixmap.fromImage( self.frame)
        item = QGraphicsPixmapItem(pix)
        scene = QGraphicsScene()
        scene.addItem(item)
        self.graphicsView.setGeometry(QtCore.QRect(350-w/2, 270-h/2, w+5, h+5))
        self.graphicsView.setScene(scene)


    def mousePress(self,event):
        self.flag = True
        self.x0 = event.pos().x()
        self.y0 = event.pos().y()
        print("開始 :　",self.x0,self.y0)
    def mouseRelease(self,event):
        self.flag = False
        self.x1 = event.x()
        self.y1 = event.y()
        #print("開始 :　",self.x0,self.y0)
        # 設定 超出畫布 就最大最小直
        #
        print("結束 :　",self.x1,self.y1)
        if self.frame != []:
            if self.mode == 'Define'  : # 要設定 是左鍵 不然右鍵也有
                if event.button() ==  1 :
                    img = self.qimg2cv(self.before_define_img)
                    self.img_x,  self.img_y , _= img.shape
                    self.norm_x = self.x1/self.img_y
                    self.norm_y = self.y1/self.img_x

                    self.to_txt(self.image_paths[self.pic_num])
                    #print(self.txt)
                    f = open(self.open_txt+'.txt')
                    for line in f.readlines():
                        #line = line.split(' ')
                        line = list(map(float, line.split(' ')))
                        xmin = line[1]-line[3]/2 
                        ymin = line[2]-line[4]/2 
                        xmax = line[1]+line[3]/2 
                        ymax = line[2]+line[4]/2 
                        if xmax > self.norm_x > xmin and ymax > self.norm_y > ymin: 
                            self.text = ''
                            self.order += 1
                            self.define_list.append([self.order,line[0],line[1],line[2],line[3],line[4]]) # 順序 , 種類 , x center ,y center , w , h 
                            for a,b,txt_x,txt_y,txt_w,txt_h in self.define_list :
                                self.text+=(str(round(txt_x,4))+" "+str(round(txt_y,4))+' Order :　'+str(a)+'\n')
                                self.show_text.setText(self.text)
                                cv2.putText(img, str(a), (int((txt_x-txt_w/4)*self.img_y),int((txt_y+txt_h/4)*self.img_x)), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2, cv2.LINE_AA)
                            self.show_text.setText(self.text)
                            cv2.imwrite('./yoma_data/temp/temp.jpg',img)
                            self.show_newImage('./yoma_data/temp/temp.jpg')
                    f.close
                elif event.button()==2 :
                    self.yoma_return()
            elif (self.mode == 'NewLabel' and self.label_flag ==True)  : ####newlabel
                img = self.qimg2cv(self.before_define_img)
                print("NewLabel")
                self.img_x,  self.img_y , _= img.shape
                self.norm_x = self.x1/self.img_y
                self.norm_y = self.y1/self.img_x

                self.to_txt(self.image_paths[self.pic_num])

                self.xx = sorted([self.x0,self.x1])
                self.yy = sorted([self.y0,self.y1])
                self.xyxy = [self.xx[0],self.yy[0],self.xx[1],self.yy[1]]
                self.temp_n = 0
                if event.button() ==  1 :
                    if (abs(self.x1-self.x0)>10 or abs(self.y1-self.y0)>10) == 1 : #不能接受用點同個地方
                        self.cv_pltxy=self.xyxy # 0 = classes hole
                        self.xywh = xyxy2xywh_transfer(self.cv_pltxy,[self.img_y,self.img_x],'xyxy2xywh')
                        self.new_label.append([0,self.xywh[0],self.xywh[1],self.xywh[2],self.xywh[3]])
                        cv2.imwrite('yoma_data/temp/temp.jpg',img) # 目前沒覆蓋 所以框框不會被保留
                        self.showImage('label')
                elif event.button() ==  2 :
                    for self.label_c,self.label_x,self.label_y,self.label_w,self.label_h in self.new_label:
                        xmin = self.label_x-self.label_w/2 
                        ymin = self.label_y-self.label_h/2 
                        xmax = self.label_x+self.label_w/2 
                        ymax = self.label_y+self.label_h/2 
                        if xmax > self.norm_x > xmin and ymax > self.norm_y > ymin: 
                            print(self.norm_x,self.norm_y)
                            # print(xmax,xmin,ymax,ymin)
                            print(self.new_label[self.temp_n])
                            self.new_label.pop(self.temp_n)
                        self.temp_n += 1 
                    cv2.imwrite('yoma_data/temp/temp.jpg',img) # 
                    self.showImage('label')
                # 按下label >> save 弄mode =label 就label_save 彈出視窗問確定覆蓋檔案嗎
                # 如果確定 就覆蓋 exp 的txt 
                # 功能 如果可以 固定圖片是一開始detect的 每個動作都會重新plot一次 並顯示 透過更改label list去更新圖片 OK 
                # 希望可以刪除label 設定 (self.mode == 'newlabel' and self.label_flag ==True  的右鍵
                # 會像define 那樣保存點到的地方 配對有在label list裡面 就刪除 重新畫圖
                #  目前有刪除  OK
                #
                ##############################################test 
            elif self.mode == 'NULL' and event.button() == 1:
                reply  = self.warning() 
                if reply == 'Define':
                    self.define()
                elif reply == 'NewLabel':
                    self.newlabel()
        else :
            QMessageBox.warning(self, "Warning", "Please Select A Photo")

    def save(self):
        #利用line Edit控件对象text()函数获取界面输入
        if self.frame == []:
            QMessageBox.warning(self, "Warning", "Please Select A Photo")
        elif self.mode == 'Define' and self.define_list != []:
            save_name, ok = QInputDialog.getText(self, 'Save File  ', 'File Name：')
            # 完善要加入如果名稱重複問要不要覆蓋檔案
            if ok and save_name:
                if self.folder_path == '':
                    self.final = './yolov7/Eclatorq/images/' + os.path.split(str(self.image_paths[self.pic_num]))[1]
                else :
                    self.final = './yolov7/Eclatorq/images/' + os.path.split(str(self.image_paths[self.pic_num]))[1]#self.folder_path +'/'+ os.path.split(str(self.image_paths[self.pic_num]))[1]
                # print(self.final)
                # self.image_paths = getImagesInDir(self.ttest )
                self.frame = QImage(self.final)
                # self.frame = QImage(self.image_paths[self.pic_num])

                img = self.qimg2cv(self.frame) ##  被改過的照片 有紅色順序
                self.img_x,  self.img_y , _= img.shape
                #把程式改成 抓原本照片 用os.path.split(str(self.image_paths[self.pic_num]))[1] 抓檔案名稱 + 路徑 開啟Eclatorq 的 0.png
                self.define_list ##label _list  # 順序 , 種類 , x center ,y center , w , h 

                for a,b,txt_x,txt_y,txt_w,txt_h in self.define_list :
                    from utils.plots import plot_one_box
                    xyxy = [(txt_x-txt_w/2)*self.img_y ,(txt_y-txt_h/2)*self.img_x , (txt_x+txt_w/2)*self.img_y , (txt_y+txt_h/2)*self.img_x]
                    # print(xyxy)
                    # im0 = cv2.resize(im0, (640, 480), interpolation=cv2.INTER_AREA)
                    plot_one_box(xyxy, img, label=str(a), color=[255,0,0], line_thickness=1) #label 
                np.savetxt(self.write_label+save_name+".txt", np.asarray(self.define_list), delimiter=" ",fmt='%1.3f')
                    #wirte final txt to  self.write_label


                cv2.imwrite('Eclatorq/sop/'+save_name+'.jpg',img)
                self.show_newImage(self.write_img+save_name+'.jpg')
                self.define_list = []
                self.next()

            # QMessageBox.warning(self, "Warning", "Define The Order")
        elif self.mode == 'NewLabel' and len(self.new_label) > 0:
            reply  = QMessageBox.warning(self, 'Alert',"This Operating Will OverWrite The File !",QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                np.savetxt(self.open_label+self.img_name+".txt", np.asarray(self.new_label), delimiter=" ",fmt='%1.6f') 
                self.mode = 'NULL'
                self.open_dir_Button.setEnabled(True)
            # self.next()
        else :  
            print("保存失敗")
    def to_txt (self,jpg_dir):
        
        self.img_path = os.path.dirname(str(jpg_dir)) #目前沒用

        self.open_label = str( self.classify_result_dir) + "/labels/"
        self.img_name = os.path.splitext(os.path.basename(jpg_dir))[0]
        self.write_img = "./Eclatorq/sop/"
        self.write_label = "./Eclatorq/sop/labels/"

        self.open_txt = self.open_label +self.img_name # D:/pycharm/yolov7/runs/detect/exp25/label/0
        # print(self.open_txt)
        self.write_txt = self.write_label +self.img_name # D:/pycharm/yolov7/Eclatorq/sop/label/0
        # print(self.write_txt)

   
    def define (self):
        self.mode = 'Define' 
        self.define_list = []
        self.order = 0 
        self.text = ''
        self.before_define_img = self.frame # = QImage(self.image_paths[self.pic_num])
        self.disp_text.setText('Mode ：'+str(self.mode)+'  File Name：'+str(self.img_name))
    def newlabel (self): 
        self.disp_text.setText('Mode ：'+str(self.mode)+'  File Name：'+str(self.img_name))
        self.mode = 'NewLabel' 
        self.order = 0 
        if self.label_flag == True:
            self.label_flag = False
            self.mode = 'NULL' 
            # self.open_dir_Button.setEnabled(True)
            # np.savetxt(self.write_label+save_name+".txt", np.asarray(self.new_label), delimiter=" ",fmt='%1.6f')
        elif self.label_flag == False:
            self.label_flag = True
            self.cv_pltxy = []
            self.new_label = []
            f = open(self.open_txt+'.txt')
            for line in f.readlines():
                line = list(map(float, line.split(' ')))
                self.new_label.append([line[0],line[1],line[2],line[3],line[4]]) #  種類 , x center ,y center , w , h 
            self.buf_label_len = len(self.new_label)
            # self.open_dir_Button.setEnabled(False)
        print(self.label_flag)
        self.before_define_img = self.frame # = QImage(self.image_paths[self.pic_num])

        #print(img.shape)
    def qimg2cv(self,q_img):
        q_img.save('temp.png', 'png')
        img = cv2.imread('temp.png')
        return img
    def yoma_return(self):
        if self.frame == []:
            QMessageBox.warning(self, "Warning", "Please Select A Photo")
        elif self.mode == 'Define' and self.define_list == []:
            QMessageBox.warning(self, "Warning", "Define The Order")
        if (self.mode == 'Define' ) :
            self.text = ''
            self.order = self.order-1
            img = self.qimg2cv(self.before_define_img)
            self.img_x,  self.img_y , _= img.shape
            print(self.define_list)
            if self.order >= 0:
                self.define_list.pop(self.order)
                for a,b,txt_x,txt_y,txt_w,txt_h in self.define_list :
                    self.text+=(str(round(txt_x,4))+" "+str(round(txt_y,4))+' Order :　'+str(a)+'\n')
                    cv2.putText(img, str(a), (int((txt_x-txt_w/4)*self.img_y),int((txt_y+txt_h/4)*self.img_x)), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2, cv2.LINE_AA)

                self.show_text.setText(self.text)
                #cv2.imwrite('yoma_data/temp/temp.jpg',img)
                #self.show_newImage('D:/pycharm/yolov7/yoma_data/temp/temp.jpg')
            else :
                self.order = 0
                self.show_text.setText(self.text)
            cv2.imwrite('./yoma_data/temp/temp.jpg',img)
            self.show_newImage('./yoma_data/temp/temp.jpg')
        # elif (self.mode == 'newlabel' and self.label_flag ==True) :
        #     pass

        #print(self.define_list)
    def open_folder(self):
        # folder_path = QFileDialog.getExistingDirectory(self,
        #           "Open folder",
        #           "./Eclatorq")                 # start path
        self.folder_path = './Eclatorq/images'
        print(self.folder_path)

        # self.pic_g.setTitle(str(self.folder_path))

        self.classify()
        # self.showImage()
    def next(self):
        if (self.mode == 'Define' and len(self.define_list) > 0) or (self.mode == 'NewLabel' and (len(self.cv_pltxy) > 0 or len(self.new_label) != self.buf_label_len)) :
            reply  = QMessageBox.question(self, 'Message'," Save or Discard ",QMessageBox.Save | QMessageBox.Discard, QMessageBox.Discard)
            if reply == QMessageBox.Save:
                self.save()
        self.pic_num+=1
        self.mode = 'NULL'
        self.define_list = []
        self.showImage()
        self.order = -1
        self.label_flag = False

    def last(self):
        if (self.mode == 'Define' and len(self.define_list) > 0) or (self.mode == 'NewLabel' and (len(self.cv_pltxy) > 0 or len(self.new_label) != self.buf_label_len)) :
            reply  = QMessageBox.question(self, 'Message'," Save or Discard ",QMessageBox.Save | QMessageBox.Discard, QMessageBox.Discard)
            if reply == QMessageBox.Save:
                self.save()
        self.pic_num-=1
        self.mode = 'NULL'
        self.define_list = []
        self.showImage()
        self.order = -1
        self.label_flag = False

    def warning(self):
        # cb = QCheckBox('目前沒用') 要查怎麼用
        msgBox = QMessageBox()
        msgBox.setWindowTitle('Warning')
        msgBox.setIcon(QMessageBox.Warning)
        msgBox.setText('Please Select Operational Mode')
        msgBox.setInformativeText('Define or NewLabel ?')
        # self.button_list = ['save','nosave']
        a = msgBox.addButton('NewLabel', QMessageBox.RejectRole)
        b = msgBox.addButton('Define', QMessageBox.AcceptRole)
        c = msgBox.addButton('None',QMessageBox.AcceptRole)
        # Cancel = msgBox.addButton('不保存', QMessageBox.DestructiveRole)
        msgBox.setDefaultButton(a)
        # msgBox.setCheckBox(cb)
        msgBox.exec()
        self.testmb_text = msgBox.clickedButton().text()   # 取得點擊的按鈕文字
        print(self.testmb_text)
        return (self.testmb_text)
        # 可以自訂義告視窗 但先用預設的就好


    # def showImage(self):
    #     height, width, channel = self.img.shape
    #     bytesPerline = 3 * width
    #     self.qImg = QImage(self.img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()

    #     self.label.setPixmap(QPixmap.fromImage(self.qImg))

def getImagesInDir(dir_path):
    image_list = []
    for filename in glob.glob(dir_path + '/*.png'):
        image_list.append(filename)
    return image_list

if __name__ == "__main__":
    #固定的，PyQt5程序都需要QApplication对象。sys.argv是命令行参数列表，确保程序可以双击运行
    app = QApplication(sys.argv)
    #初始化
    myWin = MyMainForm()
    test_classify = classify()

    #将窗口控件显示在屏幕上
    myWin.show()
    #程序运行，sys.exit方法确保程序完整退出。
    sys.exit(app.exec_())

# def mouseReleaseEvent(self,event):
    #     self.flag = False
    #     self.x1 = event.x()
    #     self.y1 = event.y()
    #     #print("開始 :　",self.x0,self.y0)
    #     print("結束 :　",self.x1,self.y1)
    #     if self.mode == 'define'  :
    #         img = self.qimg2cv(self.before_define_img)
    #         self.img_x,  self.img_y , _= img.shape
    #         self.norm_x = self.x1/self.img_y
    #         self.norm_y = self.y1/self.img_x
    #         if self.norm_x <=1 and self.norm_x <=1: 
    #             self.text = ''
    #             self.order += 1
    #             self.define_list.append([self.norm_x,self.norm_y,self.order])
    #             for a,b,c in self.define_list :
    #                 self.text+=(str(round(a,4))+" "+str(round(b,4))+' 順序 :　'+str(c)+'\n')
    #                 cv2.putText(img, str(c), (int(a*self.img_y),int(b*self.img_x)), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2, cv2.LINE_AA)
    #             self.show_text.setText(self.text)
    #             cv2.imwrite('yoma_data/temp/temp.jpg',img)
    #             self.show_newImage('D:/pycharm/yolov7/yoma_data/temp/temp.jpg')
    #     else :
    #         self.show_text.setText('plz press define')    