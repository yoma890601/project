
#### 1005 沒有tttt 那種爛參數  但可以
####
import os
import argparse
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization


# vram=1024*3
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=vram)])
#   except RuntimeError as e:
#     print(e)
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box ,plot_only_text,plot_one_box_notext
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from qreader import QReader
import open3d as o3d
import copy
qreader_reader = QReader(model_size = 'l',min_confidence = 0.5, reencode_to = 'cp65001')
from pascal_voc_writer import Writer
import DIY_ICP 
from icp import icp
import joblib

import xgboost as xgb
#          DATA_DIR = "./Eclatorq/sop/type"  # model 0 用10種 model 1 用5種
                # save_path = './yoma_data/'+save_name+'/temp'  # img.jpg

                # txt_path = './yoma_data/'+save_name +'/labels/temp'

                # new_img_path = './yoma_data/Eclatorq/input_images/'+str(datetime+'_'+p.stem)+ ('' if dataset.mode == 'image' else f'_{frame}') 
                # new_one_path = './yoma_data/Eclatorq/input_voc/' +str(datetime+'_'+p.stem)+ ('' if dataset.mode == 'image' else f'_{frame}') 

                    # path = './yoma_data'+"/order.txt"
 
                    # sop_dir = './Eclatorq/sop/labels/'+str(classs)+".txt"

# source, weights, view_img, save_txt, imgsz, trace ,classification_type= opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace , opt.classification_type
# print(source, weights, view_img, save_txt, imgsz, trace)
def get_obj_xyxy(temp_xmin,temp_ymin,temp_xmax,temp_ymax,temp_x,temp_y,img_size,mode = 1):
    img_y,  img_x , _= img_size
    obj_w= 0
    obj_h= 0
    objxmin= 0
    objymin= 0
    objxmax= 0
    objymax = 0
    if len(temp_xmin) > 0 and mode >0:
        objxmin = min(temp_xmin)*img_x
        objymin = min(temp_ymin)*img_y
        objxmax = max(temp_xmax)*img_x
        objymax = max(temp_ymax)*img_y

        # objxmin=yolo_obj_xyxy[0]
        # objymin=yolo_obj_xyxy[1]
        # objxmax=yolo_obj_xyxy[2]
        # objymax=yolo_obj_xyxy[3]
        obj_w = (objxmax - objxmin)
        obj_h = (objymax - objymin)
        obj_xyxy = [objxmin,objymin,objxmax,objymax] 
    if len(temp_x) >4 and mode >1 :
        # print(temp_x)
        nobjxmin = min(temp_x)
        nobjymin = min(temp_y)
        nobjxmax = max(temp_x)
        nobjymax = max(temp_y)
        # print(temp_y)
        nobjymid =  (nobjymax + nobjymin) /2 
        nobjxmid =  (nobjxmax + nobjxmin) /2 
        # print(sorted(temp_x))
        nobj_x4 = sorted(temp_x)[-2]
        nobj_x3 = sorted(temp_x)[-1]
        nobj_x2 = sorted(temp_x)[1]
        nobj_x1 = sorted(temp_x)[0]
        # print(nobj_x4,nobj_x3)
        # print(nobj_x2,nobj_x1)
        # print(temp_x)
        # print(temp_y)
        nobj_y4 = nobjymax
        btempy =  list(filter(lambda x: x > nobjymid, temp_y))
        stempy = list(filter(lambda x: x < nobjymid, temp_y))
        nobj_y3 = min(btempy)
        nobj_y2 = max(stempy)
        nobj_y1 = nobjymin

        y1find_x = temp_x[temp_y.index(nobj_y1)]
        y2find_x = temp_x[temp_y.index(nobj_y2)]
        y3find_x = temp_x[temp_y.index(nobj_y3)]
        y4find_x = temp_x[temp_y.index(nobj_y4)]

        x1find_y = temp_y[temp_x.index(nobj_x1)]
        x2find_y = temp_y[temp_x.index(nobj_x2)]
        x3find_y = temp_y[temp_x.index(nobj_x3)]
        x4find_y = temp_y[temp_x.index(nobj_x4)]
        pad = int((((y4find_x-y2find_x)**2+(nobj_y4-nobj_y2)**2))**0.5/2)
        temp_fx = [y1find_x,y2find_x,y3find_x,y4find_x]
        buf_temp_fx = [y1find_x,y2find_x,y3find_x,y4find_x]

        x_max_index =  buf_temp_fx.index(max(temp_fx))
        temp_fx.pop(x_max_index)
        x_max2_index = buf_temp_fx.index(max(temp_fx))


        for i in range(len(buf_temp_fx)):
            if i == x_max2_index or i == x_max_index:
                buf_temp_fx[i]=buf_temp_fx[i]+pad
            else:
                buf_temp_fx[i]=buf_temp_fx[i]-pad 

        temp_fy = [x1find_y,x2find_y,x3find_y,x4find_y]
        buf_temp_fy = [x1find_y,x2find_y,x3find_y,x4find_y]

        y_max_index =  buf_temp_fy.index(max(temp_fy))
        temp_fy.pop(y_max_index)
        y_max2_index = buf_temp_fy.index(max(temp_fy))


        for i in range(len(buf_temp_fy)):
            if i == y_max2_index or i == y_max_index:
                buf_temp_fy[i]=buf_temp_fy[i]+pad
            else:
                buf_temp_fy[i]=buf_temp_fy[i]-pad 
        y1find_x,y2find_x,y3find_x,y4find_x= buf_temp_fx

        x1find_y,x2find_y,x3find_y,x4find_y= buf_temp_fy

        y1p = [int(y1find_x),int(nobj_y1)-pad]
        y2p = [int(y2find_x),int(nobj_y2)-pad]
        y3p = [int(y3find_x),int(nobj_y3)+pad]
        y4p = [int(y4find_x),int(nobj_y4)+pad]

        x1p = [int(nobj_x1)-pad,int(x1find_y)]
        x2p = [int(nobj_x2)-pad,int(x2find_y)]
        x3p = [int(nobj_x3)+pad,int(x3find_y)]
        x4p = [int(nobj_x4)+pad,int(x4find_y)]
    return obj_w,obj_h,objxmin,objymin,objxmax,objymax
def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)
def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)
def tnet(inputs, num_features):
    # Initalise bias as the indentity matrix
    bias = tf.keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(num_features * num_features, kernel_initializer="zeros", bias_initializer=bias,
                     activity_regularizer=reg, )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])
class OrthogonalRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))
def load_dataset(DATA_DIR):
    print("load")
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_names = []

    #return train_name
    save_name_p= "yolo_train_points.npy"
    save_name_l= "yolo_train_labels.npy"
    temp = os.path.join(DATA_DIR, "*")
    folders = glob.glob(temp)
    for i, folder in enumerate(folders):
        temp = os.path.basename(folder)
        print("processing class: {}".format(temp))
        class_names.append(temp)
        train_points = np.load(save_name_p, allow_pickle=True)
        train_labels = np.load(save_name_l, allow_pickle=True)
    class_names = np.array(class_names)
    return class_names,train_points,train_labels   

def load_test(www):
        DATA_DIR = "./Eclatorq/sop/type"  # model 0 用10種 model 1 用5種

        class_names,train_points,train_labels = load_dataset(DATA_DIR)
        print(class_names)

        model_weights_name = www
        inputs = tf.keras.Input(shape=(20, 3))
        x = tnet(inputs, 3)
        x = conv_bn(x, 32)
        x = conv_bn(x, 32)
        x = tnet(x, 32)
        x = conv_bn(x, 32)
        x = conv_bn(x, 64)
        x = conv_bn(x, 256)
        x = layers.GlobalMaxPooling1D()(x)
        x = dense_bn(x, 128)
        x = layers.Dropout(0.3)(x)
        x = dense_bn(x, 64)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(len(class_names), activation="softmax")(x)

        kmodel = tf.keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        kmodel.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["sparse_categorical_accuracy"], )
        data_time=time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(os.path.getmtime ("keras.h5"))) # 日期格式完整>>"%Y-%m-%d %H:%M:%S"
        print("keras.h5 train_time : ",data_time )
        #這邊要判斷 load 的shap跟上面dataset的有一樣嗎
        kmodel.load_weights(model_weights_name)
        return class_names,kmodel
def del_file(path):
    for i in os.listdir(path):
        file_data = path + "\\" + i
        os.remove(file_data)
class classify():
    def product_testing_data(save_img=False,opt = None):
        classes = ['hole','obj']
        category_id_to_name = {0: classes[0], 1: classes[1]}
        source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
        save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
        save_txt = 1
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))
        opt.name = 'temp'
       
        # 應該要加把temp刪掉
        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if trace:
            model = TracedModel(model, device, opt.img_size)
        if half:
            model.half()  # to FP16
        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            cudnn.benchmark = True
            dataset = LoadImages(source, img_size=imgsz, stride=stride)
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1
        t0 = time.time()
        temp_save_label=[]
        last_save = 0
        num = 0

        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=opt.augment)[0]
            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=opt.augment)[0]
            t2 = time_synchronized()
            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t3 = time_synchronized()
            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path

                # txt_path = str(save_dir /  p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    # Write results
                    i = 0
                img_y,  img_x , _= im0.shape # h w 

                temp_xmin=[]
                temp_ymin=[]
                temp_xmax=[]
                temp_ymax=[]
                obj_xyxy=[]
                for *xyxyo, conf, cls in reversed(det): #改成xyxyo 因為我要用xyxy
                    if int(cls) == 0:
                        xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        xmin = xywh[0]-xywh[2]/2
                        ymin = xywh[1]-xywh[3]/2
                        xmax = xywh[0]+xywh[2]/2
                        ymax = xywh[1]+xywh[3]/2
                        temp_xmin.append(xmin)
                        temp_ymin.append(ymin)
                        temp_xmax.append(xmax)
                        temp_ymax.append(ymax)
                    #這邊只為了獲取obj大小 
                    # plot_one_box_notext(xyxyo, im0, label="", color=colors[int(cls)], line_thickness=3)
                if len(temp_xmin) > 0 :
                    objxmin = min(temp_xmin)*img_x
                    objymin = min(temp_ymin)*img_y
                    objxmax = max(temp_xmax)*img_x
                    objymax = max(temp_ymax)*img_y
                    nw = (objxmax - objxmin)#物體的w
                    nh = (objymax - objymin)
                    obj_xyxy = [objxmin,objymin,objxmax,objymax] 
                    # plot_one_box(obj_xyxy, im0, label="obj", color=[255,0,0], line_thickness=1)

                key = cv2.waitKey(1)
                num+=1
                label_list=[]
                voc_list=[]
                normal_label_list = list()
                imm=im0.copy()
                for *xyxyo, conf, cls in reversed(det):
                    if int(cls) == 0:
                        label = f'{names[int(cls)]} {conf:.2f}'    

                        xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # cv2.circle(im0, [int(xywh[0]*img_x),int( xywh[1]*img_y)], 5, (0, 255, 0), -1) #green 辨識中心點在哪
                        label = f'{names[int(cls)]} {conf:.2f}'    
                        plot_one_box(xyxyo, im0, label=label, color=colors[int(cls)], line_thickness=1)
                        label_list.append([0,xywh[0],xywh[1],xywh[2],xywh[3]])
                        voc_list.append([0,xywh[0],xywh[1],xywh[2],xywh[3]])

                        xyxy = [(xywh[0]-xywh[2]/2)*img_x-objxmin ,(xywh[1]-xywh[3]/2)*img_y-objymin, (xywh[0]+xywh[2]/2)*img_x -objxmin , (xywh[1]+xywh[3]/2)*img_y-objymin]
                        # plot_one_box(xyxy, im0, label="obj", color=[255,0,0], line_thickness=1)
                        xywh = xyxy2xywh_transfer(xyxy,[nw,nh],"xyxy2xywh")
                        # print(xywh,[nw,nh],'line 626')
                        normal_label_list.append([0,xywh[0],xywh[1],xywh[2],xywh[3]])
                    if int(cls) == 1:
                        label = f'{names[int(cls)]} {conf:.2f}'  
                        xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # cv2.circle(im0, [int(xywh[0]*img_x),int( xywh[1]*img_y)], 5, (0, 255, 0), -1) #green 辨識中心點在哪
                        voc_list.append([1,xywh[0],xywh[1],xywh[2],xywh[3]])

                        plot_one_box(xyxyo, im0, label=label, color=colors[int(cls)], line_thickness=1)
                        xyxy = [(xywh[0]-xywh[2]/2)*img_x-objxmin ,(xywh[1]-xywh[3]/2)*img_y-objymin, (xywh[0]+xywh[2]/2)*img_x-objxmin , (xywh[1]+xywh[3]/2)*img_y-objymin]
                        # plot_one_box(xyxy, im0, label="obj", color=[255,0,0], line_thickness=1)
                        xywh = xyxy2xywh_transfer(xyxy,[nw,nh],"xyxy2xywh")

                        # normal_label_list.append([1,xywh[0],xywh[1],xywh[2],xywh[3]]) # label  1  obj 

                temp_save_label.append(normal_label_list) # normal_label_list for keras pc 
                before_s_num = 500
                after_s_num = 5
                obj_xywh=[]
                if len(temp_save_label)>= before_s_num :
                    print("done")
                    temp_save_label.pop(0)
                    if key == 13:
                        print(key)
                        print(len(temp_save_label))
                        type_file_mkdir_name = str('./Eclatorq/sop/testdata/'+"TYPE C")
                        print(type_file_mkdir_name)
                        if os.path.exists(type_file_mkdir_name):
                            type_file_mkdir_name = type_file_mkdir_name+'new'
                        os.makedirs(type_file_mkdir_name)
                        for i in range (len(temp_save_label)):
                            print(i)
                            temp_name = type_file_mkdir_name+'/'+str(i)+'.txt'
                            np.savetxt(temp_name, np.asarray(temp_save_label[i]), delimiter=" ",fmt='%1.3f')#+"/1"+".txt"


                if view_img:
                    # print("video")

                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond
    def hole_detect(save_img=False,opt = None):

        source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
        print(source, weights, view_img, save_txt, imgsz, trace)
        save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
        save_txt = 1
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        save_name = os.path.split(save_dir)[1]
        print(save_name)
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()

        device = select_device(opt.device)
        # gpu
        # self.device = torch.device('cuda:0') 打包可能要改

        # # 如果只有cpu的话，就改成
        # # self.device = torch.device('cpu')

        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        if trace:
            model = TracedModel(model, device, opt.img_size)

        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            cudnn.benchmark = True
            dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=opt.augment)[0]

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=opt.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t3 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections

            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                new_img_path = 'D:/yy/image/' +str(save_name+'_'+p.stem)+ ('' if dataset.mode == 'image' else f'_{frame}') +'.jpg'# img.jpg

                new_one_path = 'D:/yy/labels/' +str(save_name+'_'+p.stem)+ ('' if dataset.mode == 'image' else f'_{frame}') # img.txt

                # txt_path = str(save_dir /  p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                img_y,  img_x , _= im0.shape # h w 

                temp_xmin=[]
                temp_ymin=[]
                temp_xmax=[]
                temp_ymax=[]
                temp_x = []
                temp_y = []
                for *xyxyo, conf, cls in reversed(det): #改成xyxyo 因為我要用xyxy
                    xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    xmin = xywh[0]-xywh[2]/2
                    ymin = xywh[1]-xywh[3]/2
                    xmax = xywh[0]+xywh[2]/2
                    ymax = xywh[1]+xywh[3]/2
                    temp_xmin.append(xmin)
                    temp_ymin.append(ymin)
                    temp_xmax.append(xmax)
                    temp_ymax.append(ymax)
                    temp_x.append(xywh[0]*640)
                    temp_y.append(xywh[1]*480)

                    #這邊只為了獲取obj大小 
                    # plot_one_box_notext(xyxyo, im0, label="", color=colors[int(cls)], line_thickness=3)
                if len(temp_x) >4:
                    # nobjxmin = min(temp_x)
                    # nobjymin = min(temp_y)
                    # nobjxmax = max(temp_x)
                    # nobjymax = max(temp_y)
                    # # print(temp_y)
                    # nobjymid =  (nobjymax + nobjymin) /2 
                    # nobjxmid =  (nobjxmax + nobjxmin) /2 
                    # # print(sorted(temp_x))
                    # nobj_x4 = sorted(temp_x)[-2]
                    # nobj_x3 = sorted(temp_x)[-1]
                    # nobj_x2 = sorted(temp_x)[1]
                    # nobj_x1 = sorted(temp_x)[0]
                    # # print(nobj_x4,nobj_x3)
                    # # print(nobj_x2,nobj_x1)
                    # # print(temp_x)
                    # # print(temp_y)
                    # nobj_y4 = nobjymax
                    # btempy =  list(filter(lambda x: x > nobjymid, temp_y))
                    # stempy = list(filter(lambda x: x < nobjymid, temp_y))
                    # nobj_y3 = min(btempy)
                    # nobj_y2 = max(stempy)
                    # nobj_y1 = nobjymin

                    # y1find_x = temp_x[temp_y.index(nobj_y1)]
                    # y2find_x = temp_x[temp_y.index(nobj_y2)]
                    # y3find_x = temp_x[temp_y.index(nobj_y3)]
                    # y4find_x = temp_x[temp_y.index(nobj_y4)]

                    # x1find_y = temp_y[temp_x.index(nobj_x1)]
                    # x2find_y = temp_y[temp_x.index(nobj_x2)]
                    # x3find_y = temp_y[temp_x.index(nobj_x3)]
                    # x4find_y = temp_y[temp_x.index(nobj_x4)]
                    # pad = int((((y4find_x-y2find_x)**2+(nobj_y4-nobj_y2)**2))**0.5/2)
                    # temp_fx = [y1find_x,y2find_x,y3find_x,y4find_x]
                    # buf_temp_fx = [y1find_x,y2find_x,y3find_x,y4find_x]

                    # x_max_index =  buf_temp_fx.index(max(temp_fx))
                    # temp_fx.pop(x_max_index)
                    # x_max2_index = buf_temp_fx.index(max(temp_fx))


                    # for i in range(len(buf_temp_fx)):
                    #     if i == x_max2_index or i == x_max_index:
                    #         buf_temp_fx[i]=buf_temp_fx[i]+pad
                    #     else:
                    #         buf_temp_fx[i]=buf_temp_fx[i]-pad 

                    # temp_fy = [x1find_y,x2find_y,x3find_y,x4find_y]
                    # buf_temp_fy = [x1find_y,x2find_y,x3find_y,x4find_y]

                    # y_max_index =  buf_temp_fy.index(max(temp_fy))
                    # temp_fy.pop(y_max_index)
                    # y_max2_index = buf_temp_fy.index(max(temp_fy))


                    # for i in range(len(buf_temp_fy)):
                    #     if i == y_max2_index or i == y_max_index:
                    #         buf_temp_fy[i]=buf_temp_fy[i]+pad
                    #     else:
                    #         buf_temp_fy[i]=buf_temp_fy[i]-pad 
                    # y1find_x,y2find_x,y3find_x,y4find_x= buf_temp_fx

                    # x1find_y,x2find_y,x3find_y,x4find_y= buf_temp_fy

                    # y1p = [int(y1find_x),int(nobj_y1)-pad]
                    # y2p = [int(y2find_x),int(nobj_y2)-pad]
                    # y3p = [int(y3find_x),int(nobj_y3)+pad]
                    # y4p = [int(y4find_x),int(nobj_y4)+pad]

                    # x1p = [int(nobj_x1)-pad,int(x1find_y)]
                    # x2p = [int(nobj_x2)-pad,int(x2find_y)]
                    # x3p = [int(nobj_x3)+pad,int(x3find_y)]
                    # x4p = [int(nobj_x4)+pad,int(x4find_y)]
                    # # print(x1p,x2p,x3p,x4p)


                    # cv2.line(im0, y1p, y2p, (0,255,0), 3) # g ## ok  y21  x 軸線
                    # cv2.line(im0, x1p, x2p, (0,0,255), 3) # r  x12
                    # cv2.line(im0, x4p, x3p, (255,0,0), 3) # b   x43
                    # cv2.line(im0, y4p, y3p, (255,255,255), 3) #white # ok y43 x軸線

                    # print(nobj_x4,nobj_x3 ,nobj_x2 ,nobj_x1 ,nobj_y4 ,nobj_y3,nobj_y2 ,nobj_y1)
                    objxmin = (min(temp_xmin)*640)*0.9
                    objymin = (min(temp_ymin)*480)*0.9
                    objxmax = (max(temp_xmax)*640)*1.1
                    objymax = (max(temp_ymax)*480)*1.1
                    nw = (objxmax - objxmin)#物體的w
                    nh = (objymax - objymin)
                    obj_xyxy = [objxmin,objymin,objxmax,objymax] 
                   
                    classes = ['obj','gg']
                    category_id_to_name = {0: classes[0], 1: classes[1]}
                    
                    #下面存檔
                    # obj_xywh = xyxy2xywh_transfer(obj_xyxy,[img_y,img_x],"xyxy2xywh")

                # if 1:  # Write to file
                #     obj_xywh = xyxy2xywh_transfer(obj_xyxy,[img_y,img_x],"xyxy2xywh")
                #     cv2.imwrite(new_img_path, im0) #改這裡 可以改存的地方
                #     line = (cls, *obj_xywh, conf) if opt.save_conf else (cls, *obj_xywh)  # label format
                #     writer = Writer(new_img_path+".jpg", img_x,  img_y)# w h 
                #     label, x_center, y_center, width, height = line
                #     x_min = int(img_y * max(float(x_center) - float(width) / 2, 0))
                #     x_max = int(img_y * min(float(x_center) + float(width) / 2, 1))
                #     y_min = int(img_x * max(float(y_center) - float(height) / 2, 0))
                #     y_max = int(img_x * min(float(y_center) + float(height) / 2, 1))

                #     writer.addObject(category_id_to_name[int(label)], x_min, y_min, x_max, y_max)
                #     writer.save(new_one_path+".xml")
                    
                #     plot_one_box(obj_xyxy, im0, label="obj", color=[255,0,0], line_thickness=1)

        ####這邊只為了獲取obj大小 
                    i = 0
                for *xyxy, conf, cls in reversed(det):
                    # i += 1
                    if conf.item() >0:#0.45 : #信心>0.45 才寫進去
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            # print(conf.item())
                            # if conf.item() >0.45 : #信心>0.45 才寫進去
                            with open(txt_path + '.txt', 'a') as f:# with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'    
                                # label = label +" "+ str(i)
                                # print(label)
                                # label = ''
                                # im0 = cv2.resize(im0, (640, 480), interpolation=cv2.INTER_AREA)

                                # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                    #im0 = cv2.resize(im0, (640, 480), interpolation=cv2.INTER_AREA)
                    # 用txt x y w h >>
                    #xmin = x-w/2 ymin = y-h/2 可得左上
                    #xmax = x+w/2 ymax = y + h/2 得右下
                    # get_clicked_position 在range中 點選保存要得
                    #新的txt 用plot 改 draw_box_on_image 畫出來

                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                # Stream results
                if view_img:
                    # print("video")
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                print("hahahahahahaaaaaaaaaaaaaaaaaa")
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0) #改這裡 可以改存的地方
                        print(f" The image with the result is saved in: {save_path}")
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)

        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')
        return  save_dir
    def type_add_detect(save_img=False,opt = None):
        classes = ['hole','obj']
        category_id_to_name = {0: classes[0], 1: classes[1]}
        source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
   
        # # print(source, weights, view_img, save_txt, imgsz, trace)
        print(view_img)
        save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
        save_txt = 1
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))
        opt.name = 'temp'
        save_dir = Path('./yoma_data/define_temp/')# Path(increment_path(Path(opt.project) / opt.name))  # increment run
        datetime=time.strftime("%Y-%m-%d.%H.%M.%S",time.localtime()) # 日期格式完整>>"%Y-%m-%d %H:%M:%S"
        print(datetime)
        # 應該要加把temp刪掉
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        save_name = os.path.split(save_dir)[1]
        print(save_dir)
        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if trace:
            model = TracedModel(model, device, opt.img_size)
        if half:
            model.half()  # to FP16
        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            cudnn.benchmark = True
            dataset = LoadImages(source, img_size=imgsz, stride=stride)
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1
        t0 = time.time()
        temp_save_label=[]
        last_save = 0
        num = 0

        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=opt.augment)[0]
            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=opt.augment)[0]
            t2 = time_synchronized()
            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t3 = time_synchronized()
            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                ## 改不會增加 會被替換掉
                # save_path = str(save_dir / p.name)  # img.jpg
                # print(save_path)
                # txt_path = str(save_dir / 'labels' / p.name)+'.txt' # img.txt
                # txt_path = str(save_dir / 'labels' / p.stem) #+ ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                # print(txt_path)
                save_path = './yoma_data/'+save_name+'/temp'  # img.jpg

                txt_path = './yoma_data/'+save_name +'/labels/temp'

                new_img_path = './yoma_data/Eclatorq/input_images/'+str(datetime+'_'+p.stem)+ ('' if dataset.mode == 'image' else f'_{frame}') 
                new_one_path = './yoma_data/Eclatorq/input_voc/' +str(datetime+'_'+p.stem)+ ('' if dataset.mode == 'image' else f'_{frame}') 


                # txt_path = str(save_dir /  p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    # Write results
                    i = 0
                img_y,  img_x , _= im0.shape # h w 

                temp_xmin=[]
                temp_ymin=[]
                temp_xmax=[]
                temp_ymax=[]
                obj_xyxy=[]
                for *xyxyo, conf, cls in reversed(det): #改成xyxyo 因為我要用xyxy
                    if int(cls) == 0:
                        xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        xmin = xywh[0]-xywh[2]/2
                        ymin = xywh[1]-xywh[3]/2
                        xmax = xywh[0]+xywh[2]/2
                        ymax = xywh[1]+xywh[3]/2
                        temp_xmin.append(xmin)
                        temp_ymin.append(ymin)
                        temp_xmax.append(xmax)
                        temp_ymax.append(ymax)
                    #這邊只為了獲取obj大小 
                    # plot_one_box_notext(xyxyo, im0, label="", color=colors[int(cls)], line_thickness=3)
                if len(temp_xmin) > 0 :
                    objxmin = min(temp_xmin)*img_x
                    objymin = min(temp_ymin)*img_y
                    objxmax = max(temp_xmax)*img_x
                    objymax = max(temp_ymax)*img_y
                    nw = (objxmax - objxmin)#物體的w
                    nh = (objymax - objymin)
                    obj_xyxy = [objxmin,objymin,objxmax,objymax] 
                    # plot_one_box(obj_xyxy, im0, label="obj", color=[255,0,0], line_thickness=1)

                key = cv2.waitKey(1)
                num+=1
                label_list=[]
                voc_list=[]
                normal_label_list = list()
                imm=im0.copy()
                for *xyxyo, conf, cls in reversed(det):
                    if int(cls) == 0:
                        label = f'{names[int(cls)]} {conf:.2f}'    

                        xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # cv2.circle(im0, [int(xywh[0]*img_x),int( xywh[1]*img_y)], 5, (0, 255, 0), -1) #green 辨識中心點在哪
                        label = f'{names[int(cls)]} {conf:.2f}'    
                        plot_one_box(xyxyo, im0, label=label, color=colors[int(cls)], line_thickness=1)
                        label_list.append([0,xywh[0],xywh[1],xywh[2],xywh[3]])
                        voc_list.append([0,xywh[0],xywh[1],xywh[2],xywh[3]])

                        xyxy = [(xywh[0]-xywh[2]/2)*img_x-objxmin ,(xywh[1]-xywh[3]/2)*img_y-objymin, (xywh[0]+xywh[2]/2)*img_x -objxmin , (xywh[1]+xywh[3]/2)*img_y-objymin]
                        # plot_one_box(xyxy, im0, label="obj", color=[255,0,0], line_thickness=1)
                        xywh = xyxy2xywh_transfer(xyxy,[nw,nh],"xyxy2xywh")
                        # print(xywh,[nw,nh],'line 626')
                        normal_label_list.append([0,xywh[0],xywh[1],xywh[2],xywh[3]])
                    if int(cls) == 1:
                        label = f'{names[int(cls)]} {conf:.2f}'  
                        xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # cv2.circle(im0, [int(xywh[0]*img_x),int( xywh[1]*img_y)], 5, (0, 255, 0), -1) #green 辨識中心點在哪
                        voc_list.append([1,xywh[0],xywh[1],xywh[2],xywh[3]])

                        plot_one_box(xyxyo, im0, label=label, color=colors[int(cls)], line_thickness=1)
                        xyxy = [(xywh[0]-xywh[2]/2)*img_x-objxmin ,(xywh[1]-xywh[3]/2)*img_y-objymin, (xywh[0]+xywh[2]/2)*img_x-objxmin , (xywh[1]+xywh[3]/2)*img_y-objymin]
                        # plot_one_box(xyxy, im0, label="obj", color=[255,0,0], line_thickness=1)
                        xywh = xyxy2xywh_transfer(xyxy,[nw,nh],"xyxy2xywh")

                        # normal_label_list.append([1,xywh[0],xywh[1],xywh[2],xywh[3]]) # label  1  obj 

                temp_save_label.append(normal_label_list) # normal_label_list for keras pc 
                before_s_num = 10
                after_s_num = 5
                obj_xywh=[]
                if len(temp_save_label)>= before_s_num and key !=115 and last_save < after_s_num:
                    cv2.putText(im0, 'Press S to Save', (int(img_x/4), int(img_y/4)), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)



                    temp_save_label.pop(0)
                    # 基本上 temp_save_label > 10 按下s後會繼續儲存
                elif len(temp_save_label)>= before_s_num and (key == 115 or (last_save > after_s_num)): # 進不去 
                    cv2.imwrite(save_path+".jpg", imm)  # 保存當下圖片(沒畫label的) 顯示在框框中# imm 是im0還沒畫圖
                    last_save +=1

                    obj_xywh = xyxy2xywh_transfer(obj_xyxy,[img_y,img_x],"xyxy2xywh")
                    if len(obj_xywh) >0 : #保護機制
                        writer = Writer(new_img_path+".jpg", img_x,  img_y)# w h 
                        cv2.imwrite(new_img_path+".jpg", imm) #yoma_data\Eclatorq yolo retrain # imm 是im0還沒畫圖
                        # print(voc_list)
                        for line in voc_list:
                            # line = (cls, *obj_xywh, conf) if opt.save_conf else (cls, *obj_xywh)  # label format
                            label, x_center, y_center, width, height = line
                            x_min = int(img_x * max(float(x_center) - float(width) / 2, 0))
                            x_max = int(img_x * min(float(x_center) + float(width) / 2, 1))
                            y_min = int(img_y * max(float(y_center) - float(height) / 2, 0))
                            y_max = int(img_y * min(float(y_center) + float(height) / 2, 1))
                            # print(category_id_to_name[int(label)], x_min, y_min, x_max, y_max)
                            writer.addObject(category_id_to_name[int(label)], x_min, y_min, x_max, y_max)
                            writer.save(new_one_path+".xml") #這邊要改設定完名稱後 存到./Eclatorq/type
                        # for file in Path(file_source).glob("randomfile.txt"):
                            # shutil.move(os.path.join(file_source, file), file_destination)
                    countdown = after_s_num -last_save
                    if countdown >0 :
                        countdown_text = " Wait "+str(countdown)+" Saving"
                        cv2.putText(im0, countdown_text, (40, 40), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)

                    if last_save > after_s_num : # after_s_num + before_s_num 張
                        # cv2.imwrite("D:/yy/ww.jpg", im0)  #保存當下圖片 顯示在框框中
                        # np.savetxt("D:/yy/ww"+".txt", np.asarray(temp_save_label[i]), delimiter=" ",fmt='%1.3f')#+"/1"+".txt"
                        # print('save temp')
                        #print(save_path)
                        cv2.imwrite(save_path+".jpg", imm)   # 保存當下圖片(沒畫label的) 顯示在框框中# imm 是im0還沒畫圖
                        # imm 是im0還沒畫圖
                        np.savetxt(txt_path+'.txt', np.asarray(label_list), delimiter=" ",fmt='%1.3f')#+"/1"+".txt"
                        cv2.putText(im0, 'Press q to Close Windows', (0, int(img_y/4)), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), 2)
                        # print('save 40 label destroyAllWindows ')
                        # 目前會一直儲存 要改一下
                        # self.stop=0
                        print(save_dir)
                        if key == 27:
                            print(key)
                            print(temp_save_label,save_dir)
                            return temp_save_label,save_path
                        # cv2.destroyAllWindows()
                    # for i in range(len(temp_save_label)):
                    # np.savetxt(DATA_DIR+"/1_"+str(num)+".txt", np.asarray(temp_save_label[i]), delimiter=" ",fmt='%1.3f')#+"/1"+".txt"

                    # np.savetxt(DATA_DIR+"/1_"+str(num)+".txt", np.asarray(label_list), delimiter=" ",fmt='%1.3f')#+"/1"+".txt"
                    # 上面 label_list  再套一層  50_label_list.append(label_list)
                    # 原本會一直替換 按下保存後 抓前30個label 然後 再抓30個 就 cv2.destroyallwindows()
                # Print time (inference + NMS)

                # Stream results
                if view_img:
                    # print("video")
                    cv2.namedWindow('Fullscreen Window', cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty('Fullscreen Window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                # if save_img:
                #     if dataset.mode == 'image':
                #         cv2.imwrite(save_path, im0) #改這裡 可以改存的地方
                #         print(f" The image with the result is saved in: {save_path}")
                #     else:  # 'video' or 'stream'
                #         if vid_path != save_path:  # new video
                #             vid_path = save_path
                #             if isinstance(vid_writer, cv2.VideoWriter):
                #                 vid_writer.release()  # release previous video writer
                #             if vid_cap:  # video
                #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
                #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                #             else:  # stream
                #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
                #                 save_path += '.mp4'
                #             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                #         vid_writer.write(im0)

        return  temp_save_label,save_dir
    def hole_class_detect_buf(save_img=False,opt = None): # 多了 model_weights_name let call class can chose .h5 file 
        source, weights, view_img, save_txt, imgsz, trace ,model_weights_name= opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace , opt.model_weights_name
        print(model_weights_name,imgsz)

        class_names,kmodel = load_test(model_weights_name) # load tf.keras.Model once !!

        RF=joblib.load('rf.model')
        xgboostModel = xgb.XGBClassifier()
        xgboostModel.load_model("xgb.json")
        print( class_names,kmodel)
        save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
        save_txt = 1
        yolo_mode = 1
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if trace:
            model = TracedModel(model, device, opt.img_size)
        if half:
            model.half()  # to FP16
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
        vid_path, vid_writer = None, None
        if webcam: #理論上進這裡 
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            cudnn.benchmark = True
            dataset = LoadImages(source, img_size=imgsz, stride=stride)
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1
        t0 = time.time()
        mid_preds = []
        buf_icp_target = []
        buf_obj_xyxy = []
        result_list = [[],[],[]]
        block_order_xy= [0,0]
        lock_order_xy = [0,0]
        #print("do once")
        try :
            path = './yoma_data'+"/order.txt"
            f = open(path, 'w')
            # indata = 1 # 初始直
            # print(indata)
            f.write(str(1))
        except Exception as e:
            print(e)
        for path, img, im0s, vid_cap in dataset:

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=opt.augment)[0]
            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=opt.augment)[0]
            t2 = time_synchronized()
            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t3 = time_synchronized()
            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                # txt_path = str(save_dir /  p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                obj_hole = []
                obj_hole2 = []

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                img_y,  img_x , _= im0.shape
                # print(img_y,  img_x)
                locktimes =1

                if yolo_mode == 1 :
                    temp_sop_list = []
                    temp_xmin=[]
                    temp_ymin=[]
                    temp_xmax=[]
                    temp_ymax=[]
                    temp_x = []
                    temp_y = []
                    for *xyxyo, conf, cls in reversed(det): #改成xyxyo 因為我要用xyxy
                        if int(cls) == 0:
                            xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh 影像的正規化
                            xmin = xywh[0]-xywh[2]/2
                            ymin = xywh[1]-xywh[3]/2
                            xmax = xywh[0]+xywh[2]/2
                            ymax = xywh[1]+xywh[3]/2
                            temp_xmin.append(xmin)
                            temp_ymin.append(ymin)
                            temp_xmax.append(xmax)
                            temp_ymax.append(ymax)
                            temp_x.append(xywh[0]*img_x)
                            temp_y.append(xywh[1]*img_y)
                        
                        if int(cls) == 1:
                            xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh 影像的正規化
                            yolo_obj_xyxy = xyxy2xywh_transfer(xywh,[img_x,img_y],"xywh2xyxy") # yolo obj xmin ymin xmax ymax
                            # print(yolo_obj_xyxy)

                    if len(temp_xmin) > 0 :
                        objxmin = min(temp_xmin)*img_x
                        objymin = min(temp_ymin)*img_y
                        objxmax = max(temp_xmax)*img_x
                        objymax = max(temp_ymax)*img_y

                        # objxmin=yolo_obj_xyxy[0]
                        # objymin=yolo_obj_xyxy[1]
                        # objxmax=yolo_obj_xyxy[2]
                        # objymax=yolo_obj_xyxy[3]
                        obj_w = (objxmax - objxmin)
                        obj_h = (objymax - objymin)
                        obj_xyxy = [objxmin,objymin,objxmax,objymax] 
                        # plot_one_box(obj_xyxy, im0, label="obj", color=[255,0,0], line_thickness=1)

                    if len(temp_x) >4:
                        # print(temp_x)
                        nobjxmin = min(temp_x)
                        nobjymin = min(temp_y)
                        nobjxmax = max(temp_x)
                        nobjymax = max(temp_y)
                        # print(temp_y)
                        nobjymid =  (nobjymax + nobjymin) /2 
                        nobjxmid =  (nobjxmax + nobjxmin) /2 
                        # print(sorted(temp_x))
                        nobj_x4 = sorted(temp_x)[-2]
                        nobj_x3 = sorted(temp_x)[-1]
                        nobj_x2 = sorted(temp_x)[1]
                        nobj_x1 = sorted(temp_x)[0]
                        # print(nobj_x4,nobj_x3)
                        # print(nobj_x2,nobj_x1)
                        # print(temp_x)
                        # print(temp_y)
                        nobj_y4 = nobjymax
                        btempy =  list(filter(lambda x: x > nobjymid, temp_y))
                        stempy = list(filter(lambda x: x < nobjymid, temp_y))
                        nobj_y3 = min(btempy)
                        nobj_y2 = max(stempy)
                        nobj_y1 = nobjymin

                        y1find_x = temp_x[temp_y.index(nobj_y1)]
                        y2find_x = temp_x[temp_y.index(nobj_y2)]
                        y3find_x = temp_x[temp_y.index(nobj_y3)]
                        y4find_x = temp_x[temp_y.index(nobj_y4)]

                        x1find_y = temp_y[temp_x.index(nobj_x1)]
                        x2find_y = temp_y[temp_x.index(nobj_x2)]
                        x3find_y = temp_y[temp_x.index(nobj_x3)]
                        x4find_y = temp_y[temp_x.index(nobj_x4)]
                        pad = int((((y4find_x-y2find_x)**2+(nobj_y4-nobj_y2)**2))**0.5/2)
                        temp_fx = [y1find_x,y2find_x,y3find_x,y4find_x]
                        buf_temp_fx = [y1find_x,y2find_x,y3find_x,y4find_x]

                        x_max_index =  buf_temp_fx.index(max(temp_fx))
                        temp_fx.pop(x_max_index)
                        x_max2_index = buf_temp_fx.index(max(temp_fx))


                        for i in range(len(buf_temp_fx)):
                            if i == x_max2_index or i == x_max_index:
                                buf_temp_fx[i]=buf_temp_fx[i]+pad
                            else:
                                buf_temp_fx[i]=buf_temp_fx[i]-pad 

                        temp_fy = [x1find_y,x2find_y,x3find_y,x4find_y]
                        buf_temp_fy = [x1find_y,x2find_y,x3find_y,x4find_y]

                        y_max_index =  buf_temp_fy.index(max(temp_fy))
                        temp_fy.pop(y_max_index)
                        y_max2_index = buf_temp_fy.index(max(temp_fy))


                        for i in range(len(buf_temp_fy)):
                            if i == y_max2_index or i == y_max_index:
                                buf_temp_fy[i]=buf_temp_fy[i]+pad
                            else:
                                buf_temp_fy[i]=buf_temp_fy[i]-pad 
                        y1find_x,y2find_x,y3find_x,y4find_x= buf_temp_fx

                        x1find_y,x2find_y,x3find_y,x4find_y= buf_temp_fy

                        y1p = [int(y1find_x),int(nobj_y1)-pad]
                        y2p = [int(y2find_x),int(nobj_y2)-pad]
                        y3p = [int(y3find_x),int(nobj_y3)+pad]
                        y4p = [int(y4find_x),int(nobj_y4)+pad]

                        x1p = [int(nobj_x1)-pad,int(x1find_y)]
                        x2p = [int(nobj_x2)-pad,int(x2find_y)]
                        x3p = [int(nobj_x3)+pad,int(x3find_y)]
                        x4p = [int(nobj_x4)+pad,int(x4find_y)]
                        # print(x1p,x2p,x3p,x4p)


                        # cv2.line(im0, y1p, y2p, (0,255,0), 3) # g ## ok  y21  x 軸線
                        # cv2.line(im0, x1p, x2p, (0,0,255), 3) # r  x12
                        # cv2.line(im0, x4p, x3p, (255,0,0), 3) # b   x43
                        # cv2.line(im0, y4p, y3p, (255,255,255), 3) #white # ok y43 x軸線
                        

                    ####這邊只為了獲取obj大小 
                    try_icp_target = []
                    
                    ############classify type qq
                    for *xyxyo, conf, cls in reversed(det): #從640 480的正規化 轉成物體大小的正規化
                        if int(cls) == 0:

                            xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh 影像的正規化
                            xyxy = [(xywh[0]-xywh[2]/2)*img_x-objxmin ,(xywh[1]-xywh[3]/2)*img_y -objymin, (xywh[0]+xywh[2]/2)*img_x-objxmin , (xywh[1]+xywh[3]/2)*img_y-objymin]
                            # try_icp_target.append([xywh[0],xywh[1],0])
                            
                            # try_icp_target.append([round((xyxy[0]+xyxy[2])/2, 3),round((xyxy[1]+xyxy[3])/2, 3),0])
                            
                            xywh = xyxy2xywh_transfer(xyxy,[obj_w,obj_h],"xyxy2xywh")#物體的比例
                            xyxy = xyxy2xywh_transfer(xywh,[640,480],"xywh2xyxy")#給type的model轉回640*480 那時候model這樣train的
                                                                                 #可以看 point cloud yolo_桌 load_data 設定
                                                                                 #以固定畫面640*480沒差 但之後化面改變會出事  所以先處理
                            try_icp_target.append([xywh[0],xywh[1]])
                            obj_pc = ([round((xyxy[0]+xyxy[2])/2, 3),round((xyxy[1]+xyxy[3])/2, 3),0]) 
                            obj_pc2 = ([round((xyxy[0]+xyxy[2])/2, 3),round((xyxy[1]+xyxy[3])/2, 3)]) 
                            obj_hole2.append(obj_pc2)
                            obj_hole.append(obj_pc)
                        if int(cls) == 1:
                            xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh 影像的正規化 
                            # 可以抓obj 的參數 x center , ycenter ,w, h

                    obj_hole = obj_hole
                    obj_hole2 = obj_hole2

                    empty = [0,0,0]
                    if len(obj_hole) < 20  :
                        for b in range(20-len(obj_hole)):
                            obj_hole2.append([0,0])
                            obj_hole.append([b,b,b])
                    obj = (np.array(obj_hole)).reshape(1,20,3)
                    obj2 = (np.array(obj_hole2)).reshape(1,40)
                    preds = kmodel.predict(obj) # Point Net  
                    topreds = np.argmax(preds, -1) ## Point Net 分類用

                    RF_topreds = RF.predict(obj2) 
                    xgb_topreds = xgboostModel.predict(obj2) 
                    print("pn: ",topreds,"RF : ",RF_topreds,"xgb : ",xgb_topreds)

                    # classs = class_names[topreds][0]
                    # 如果type model不會出錯 可以拿掉下面的 用上面那句
                    # 但保留可以增加穩定性

                    result_list[0].append(topreds[0])
                    result_list[1].append(RF_topreds[0])
                    result_list[2].append(xgb_topreds[0])
                    #list.count(obj)

                    mid_preds.append(topreds[0])
                    mid_times = 10
                    if len(mid_preds) >mid_times:
                        mid_preds.pop(0)
                    maxlabel = max(mid_preds,key=mid_preds.count)
                    classs = class_names[maxlabel]

                    # print("model_type : ",classification_type,classs)

                    cv2.putText(im0, str(classs), (40, 40), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                    sop_dir = './Eclatorq/sop/labels/'+str(classs)+".txt"
                    f = open(sop_dir)
                    try_icp_source = []
                    source_temp=[]

                    for line in f.readlines():
                        line = list(map(float, line.split(' ')))
                        temp_sop_list.append([line[0],line[1],line[2],line[3],line[4],line[5],0]) # 順序 , 種類 , x center ,y center , w , h 
                        try_icp_source.append([line[2],line[3]])
                        source_temp.append([line[2],line[3],0])

                    ############classify type 
                    ############ icp  
                    # for i in range(len(try_icp_target)):
                    #     cv2.circle(im0, [int(try_icp_target[i][0]*640),int(try_icp_target[i][1]*480)], 5, (0, 0, 255), -1) #
                    # for i in range(len(try_icp_source)):
                    #     cv2.circle(im0, [int(try_icp_source[i][0]*640),int(try_icp_source[i][1]*480)], 2, (0, 255, 0), -1) #
                    final_point =[]

                    if len(try_icp_target) != len(try_icp_source):
                        icp_mode='mine'
                    else:
                        icp_mode='mine' # 'open3d'   icp_kd'  'mine'
                    if icp_mode == 'open3d':
                        if len(try_icp_target) >0 :# try_icp_source try_icp_target
                            target = np.asarray(try_icp_target) # try_icp_target 
                            source = np.asarray(try_icp_source) #  用sop當要被改變得 #source 需要匹配的
                                                                #  辨識點為target 讓sop照著看到的方向轉變 
                            source_temp=np.asarray(source_temp)

                            newtarget = []
                            newsource = []
                            for i in range(len(try_icp_target)):
                                newtarget.append([target[i][0],target[i][1],0])
                                newtarget.append([target[i][0],target[i][1],1])
                            for i in range(len(try_icp_source)):
                                newsource.append([source[i][0],source[i][1],0])
                                newsource.append([source[i][0],source[i][1],1])

                            pcd_target = o3d.geometry.PointCloud()
                            pcd_source = o3d.geometry.PointCloud()
                            pcd_source_temp = o3d.geometry.PointCloud()
                            pcd_target.points = o3d.utility.Vector3dVector(newtarget)
                            pcd_source_temp.points = o3d.utility.Vector3dVector(newsource)
                            pcd_source_temp.points = o3d.utility.Vector3dVector(source_temp)
                            pcd_source.paint_uniform_color([1, 0, 0])    #source r色
                            pcd_target.paint_uniform_color([0, 0, 1])   #target b色
                            pcd_source_temp.paint_uniform_color([0, 1, 0]) #g
                            threshold = 0.1
                            trans_init = np.asarray([[1,0,0,0],   [0,1,0,0],   [0,0,1,0],   [0,0,0,1]]) # buf_transformation

                            reg_p2p = o3d.pipelines.registration.registration_icp(
                                pcd_source, pcd_target, threshold, trans_init,
                                o3d.pipelines.registration.TransformationEstimationPointToPoint(),o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 10000))
                            pcd_source_temp.transform(reg_p2p.transformation)
                            final_point = np.asarray(pcd_source_temp.points)
                    if icp_mode=='icp_kd':
                        if len(try_icp_target) >0 :# try_icp_source try_icp_target
                            reference_points = np.asarray(try_icp_source) # try_icp_target      # 需要被改變
                            points_to_be_aligned = np.asarray(try_icp_target)     # 被對齊 sop
                            # points_to_be_aligned += np.array([np.random.random_sample(), np.random.random_sample()])
                            transformation_history, final_point = icp(reference_points, points_to_be_aligned, verbose=True)
                            # print(final_point)
                    if icp_mode=='mine':
                        if len(try_icp_target) >0 :# try_icp_source try_icp_target
                            target = np.asarray(try_icp_target) # try_icp_target 
                            source = np.asarray(try_icp_source) #  用sop當要被改變得 #source 需要匹配的
                                                                #  辨識點為target 讓sop照著看到的方向轉變 
                            # print('target',target) # ys yolo   >> sop 轉成yolo看到的
                            # print('source',source) # ys yolo   >> sop 轉成yolo看到的

                            final_point,test_dis= DIY_ICP.Fit(source,target,show=0,show_f=0)
                            # 基本上 可以照 s_index 的順序  
                            #所以 for i in s_index : xyxy
                            #  或是改這邊   int(temp_sop_list[i][0]) 
                                #if xmax+objxmin > xywh[0]*img_x > xmin+objxmin and ymax+objymin>xywh[1]*img_y > ymin+objymin: 
                                #plot_one_box(xyxyo, im0, label=str(int(temp_sop_list[i][0])), color=[0,0,255], line_thickness=3)#colors[int(cls)]
                                    


                            # final_point, test_dis= DIY_ICP.Fit_show(source,target,show=0)
                            # print(final_point, test_dis)
                            # final_point = np.asarray(pcd_source_temp.points)
                        # icp
                    # print(len(final_point),len(temp_sop_list))
                    if min(len(temp_sop_list),len(final_point)) > 0 :
                        for i in range( min(len(temp_sop_list),len(final_point)) ):
                            temp_sop_list[i][2] = final_point[i][0]
                            temp_sop_list[i][3] = final_point[i][1]
                            
                            cv2.circle(im0, [int(final_point[i][0]*(objxmax-objxmin)+objxmin),int(final_point[i][1]*(objymax-objymin)+objymin)], 3, (0, 255, 255), -1) #red 轉換後 lock_order 點
                    ############ icp       
                    ############ 判斷出type 後給順序 
                    # tttt_xyxy0=[]
                    # tttt_text = []
                    yyy=0
                    for *xyxyo, conf, cls in reversed(det):
                        if int(cls) == 0:
                            xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            cv2.circle(im0, [int(xywh[0]*img_x),int( xywh[1]*img_y)], 4, (255, 0, 0), -1) #green 辨識中心點在哪
                            text = ""
                            find_range = 1.2
                            for i in range(len(temp_sop_list)):
                                xmin = (temp_sop_list[i][2]-temp_sop_list[i][4]/find_range)*obj_w
                                ymin = (temp_sop_list[i][3]-temp_sop_list[i][5]/find_range)*obj_h
                                xmax = (temp_sop_list[i][2]+temp_sop_list[i][4]/find_range)*obj_w
                                ymax = (temp_sop_list[i][3]+temp_sop_list[i][5]/find_range)*obj_h
                                t = [xmin+objxmin,ymin+objymin,xmax+objxmin,ymax+objymin]
                                if xmax+objxmin > xywh[0]*img_x > xmin+objxmin and ymax+objymin>xywh[1]*img_y > ymin+objymin: 
                                    # tttt_xyxy0.append(xyxyo) #把洞跟順序分別抓下來  到時候用i就可以抓到對應的
                                    # tttt_text .append(str(text))
                                    plot_one_box(xyxyo, im0, label=str(int(temp_sop_list[i][0])), color=[0,0,255], line_thickness=3)#colors[int(cls)]

                                
                        if int(cls) == 1:
                            label = f'{names[int(cls)]} {conf:.2f}'    
                            plot_one_box(xyxyo, im0, label=label, color=colors[int(cls)], line_thickness=1)
                
                    buf_transformation = np.asarray([[1,0,0,0],   [0,1,0,0],   [0,0,1,0],   [0,0,0,1]]) 
                elif yolo_mode == 2 :#鎖固###########################################################################################################################################
                    path = './yoma_data'+"/order.txt"
                    try :
                        f = open(path, 'r')
                        lock_order = f.read()
                        # print(lock_order)
                    except Exception as e:
                        print(e)
                    f.close()
                    # lock_order_xy=[0,0]
                    #如果洞數量小於sop 2個 就用之前得物體大小    
                    sop_dir = './Eclatorq/sop/labels/'+str(classs)+".txt"
                    f = open(sop_dir)
                    # temp_sop_list = []
                    temp_xmin=[]
                    temp_ymin=[]
                    temp_xmax=[]
                    temp_ymax=[]
                    temp_x = []
                    temp_y = []
                    wrench_xyxy =[0,0,0,0]

                    for *xyxyo, conf, cls in reversed(det): #改成xyxyo 因為我要用xyxy
                        if int(cls) == 0:
                            xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh 影像的正規化
                            xmin = xywh[0]-xywh[2]/2
                            ymin = xywh[1]-xywh[3]/2
                            xmax = xywh[0]+xywh[2]/2
                            ymax = xywh[1]+xywh[3]/2
                            temp_xmin.append(xmin)
                            temp_ymin.append(ymin)
                            temp_xmax.append(xmax)
                            temp_ymax.append(ymax)
                            temp_x.append(xywh[0]*img_x)
                            temp_y.append(xywh[1]*img_y)
                        
                        if int(cls) == 1:
                            xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh 影像的正規化
                            yolo_obj_xyxy = xyxy2xywh_transfer(xywh,[img_x,img_y],"xywh2xyxy") # yolo obj xmin ymin xmax ymax
                            # print(yolo_obj_xyxy)
                        if int(cls) == 2: # 板手 wrench 
                            xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh 影像的正規化
                            yolo_wrench_xyxy = xyxy2xywh_transfer(xywh,[img_x,img_y],"xywh2xyxy") # yolo obj xmin ymin xmax ymax
                            # print(yolo_wrench_xyxy)
                            wrench_xyxy =[yolo_wrench_xyxy[0],yolo_wrench_xyxy[1],yolo_wrench_xyxy[2],yolo_wrench_xyxy[3]]
                            # print(wrench_xyxy)
                            plot_one_box(wrench_xyxy, im0, label="wrench", color=[255,255,255], line_thickness=1)



                    if len(temp_xmin) > 0 :
                        # objxmin = min(temp_xmin)*img_x
                        # objymin = min(temp_ymin)*img_y
                        # objxmax = max(temp_xmax)*img_x
                        # objymax = max(temp_ymax)*img_y

                        objxmin=yolo_obj_xyxy[0]
                        objymin=yolo_obj_xyxy[1]
                        objxmax=yolo_obj_xyxy[2]
                        objymax=yolo_obj_xyxy[3]
                        obj_w = (objxmax - objxmin)
                        obj_h = (objymax - objymin)
                        obj_xyxy = [objxmin,objymin,objxmax,objymax] 

                    # lock_order_xy=[0,0]
                    #如果洞數量小於sop 2個 就用之前得物體大小    
                    sop_dir = './Eclatorq/sop/labels/'+str(classs)+".txt"
                    f = open(sop_dir)
                    try_icp_source = []
                    for line in f.readlines():
                        #line = line.split(' ')
                        line = list(map(float, line.split(' ')))
                        sop_order = int(line[0])-1
                        temp_sop_list[sop_order][2] = line[2]
                        temp_sop_list[sop_order][3] = line[3] # temp_sop_list[order]
                        temp_sop_list[sop_order][4] = line[4]
                        temp_sop_list[sop_order][5] = line[5]
                        # temp_sop_list.append([line[0],line[1],line[2],line[3],line[4],line[5]]) # 順序 , 種類 , x center ,y center , w , h 
                        try_icp_source.append([line[2],line[3]]) # (x,y,z)
                        # try_icp_source.append([line[2]*640,line[3]*480,0]) # icp 畫面版 非正規畫板

                    qrcode_xyxy =[0,0,0,0]
                    qrcode_xy =[0,0]

                    blk = np.zeros(im0.shape, np.uint8)  

                    # decoded_text = qreader_reader.detect(image=im0)
                    # if decoded_text != []:
                    #     qrcode_xyxy = decoded_text[0]['bbox_xyxy']
                    #     qrcode_xy = [(qrcode_xyxy[0]+qrcode_xyxy[2])/2,(qrcode_xyxy[1]+qrcode_xyxy[3])/2] # qrcode_center
                    #     plot_one_box(qrcode_xyxy, im0, label="qrcode", color=[255,0,0], line_thickness=1)

                    icp_target = []
                    temp_x = []
                    temp_y = []
                    for *xyxyo, conf, cls in reversed(det): #改成xyxyo 因為我要用xyxy
                        if int(cls) == 0:
                            
                            xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh 影像的正規化
                            temp_x.append(xywh[0]*img_x)
                            temp_y.append(xywh[1]*img_y)
                            xyxy = [(xywh[0]-xywh[2]/2)*img_x-objxmin ,(xywh[1]-xywh[3]/2)*img_y -objymin, (xywh[0]+xywh[2]/2)*img_x-objxmin , (xywh[1]+xywh[3]/2)*img_y-objymin]
                            if  wrench_xyxy[2] > int(xywh[0]*img_x) > wrench_xyxy[0] and wrench_xyxy[3]>int(xywh[1]*img_y) > wrench_xyxy[1]: 
                                pass
                            else :
                                cv2.circle(im0, [int(xywh[0]*img_x),int( xywh[1]*img_y)], 5, (255, 0, 0), -1) #blue 辨識中心點在哪
                                xmin = xywh[0]-xywh[2]/2
                                ymin = xywh[1]-xywh[3]/2
                                xmax = xywh[0]+xywh[2]/2
                                ymax = xywh[1]+xywh[3]/2

                                xywh = xyxy2xywh_transfer(xyxy,[obj_w,obj_h],"xyxy2xywh")
                                xyxy = xyxy2xywh_transfer(xywh,[640,480],"xywh2xyxy")
                                xyxy_center = [xywh[0],xywh[1]] # xyxy_center

                                # xyxy_center = [(xyxy[0]+xyxy[2])/2,(xyxy[1]+xyxy[3])/2,0] # icp 畫面版 非正規畫板

                                icp_target.append(xyxy_center)
                            # print(xywh[0])
                        if int(cls) == 1:
                            label = f'{names[int(cls)]}'    
                            plot_one_box(xyxyo, im0, label=label, color=colors[int(cls)], line_thickness=1)
                    if len(temp_x) > 0 :  # 可以把xmin 改掉
                        # print((len(try_icp_source) - len(temp_x)))

                        if (len(try_icp_source) - len(temp_x)) ==0: 
                            # nobjymin = min(temp_y)
                            # nobjymax = max(temp_y)
                            # nobjymid =  (nobjymax + nobjymin) /2 
                            # btempy =  list(filter(lambda x: x > nobjymid, temp_y))
                            # stempy = list(filter(lambda x: x < nobjymid, temp_y))
                            # if len(btempy) >0 and len(stempy) >0:
                            #     nobj_y4 = nobjymax
                            #     nobj_y3 = min(btempy)
                            #     nobj_y2 = max(stempy)
                            #     nobj_y1 = nobjymin
                            #     y1find_x = temp_x[temp_y.index(nobj_y1)]#*640
                            #     y2find_x = temp_x[temp_y.index(nobj_y2)]#*640  
                            #     y3find_x = temp_x[temp_y.index(nobj_y3)]#*640
                            #     y4find_x = temp_x[temp_y.index(nobj_y4)]#*640
                            #     pad = int((((y4find_x-y2find_x)**2+(nobj_y4-nobj_y2)**2))**0.5/2)
                            #     temp_fx = [y1find_x,y2find_x,y3find_x,y4find_x]
                            #     buf_temp_fx = [y1find_x,y2find_x,y3find_x,y4find_x]
                            #     x_max_index =  buf_temp_fx.index(max(temp_fx))
                            #     temp_fx.pop(x_max_index)
                            #     x_max2_index = buf_temp_fx.index(max(temp_fx))
                            #     obj_w = (max(buf_temp_fx) - min(buf_temp_fx))#物體的w
                            #     obj_h = (nobj_y4 - nobj_y1)

                            #     for i in range(len(buf_temp_fx)):
                            #         if i == x_max2_index or i == x_max_index:
                            #             buf_temp_fx[i]=buf_temp_fx[i]+pad
                            #         else:
                            #             buf_temp_fx[i]=buf_temp_fx[i]-pad 
                            #     y1find_x,y2find_x,y3find_x,y4find_x= buf_temp_fx
                            #     y1p = [int(y1find_x),int(nobj_y1)-pad]
                            #     y2p = [int(y2find_x),int(nobj_y2)-pad]
                            #     y3p = [int(y3find_x),int(nobj_y3)+pad]
                            #     y4p = [int(y4find_x),int(nobj_y4)+pad]
                            #     buf_obj_xyxy = [min(buf_temp_fx),nobj_y1-pad,max(buf_temp_fx),nobj_y4+pad] 

                            # print(y1p,y2p,y3p,y4p)
                            buf_obj_xyxy = [objxmin,objymin,objxmax,objymax] # 有這個代表正常
                            # print(buf_obj_xyxy)
                            buf_icp_target = icp_target
                            # print("1")
                            cv2.putText(im0, str(classs)+" order : "+str(lock_order), (10, 40),cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                            ww=0
                            # print((len(try_icp_source) - len(temp_x)),buf_icp_target,buf_obj_xyxy)
                        elif (len(try_icp_source) - len(temp_x)) >= 1 :#被擋住1洞 and buf_icp_target == [] and buf_icp_target != [[0,0,0]]
                            # buf_icp_target=[[0,0,0]]
                            buf_icp_target = icp_target

                            buf_obj_xyxy = [objxmin,objymin,objxmax,objymax] # 有這個代表正常
                            # print((len(try_icp_source) - len(temp_x)),buf_icp_target,buf_obj_xyxy)
                            cv2.putText(im0, str(classs)+" order : "+str(lock_order), (10, 40),cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                            ww=0
                            print(buf_obj_xyxy)
                        elif (len(try_icp_source) - len(temp_x)) >= 3 :#被擋住2洞 and buf_icp_target == [] and buf_icp_target != [[0,0,0]]
                            # buf_icp_target=[[0,0,0]]

                            buf_obj_xyxy = [objxmin,objymin,objxmax,objymax] 
                            print("請調整")
                            ww =1  #目前+ww有比較不會一直按enter 一直鎖固 但應該需要rc綠波 不然畫面一直跳
                            #目前
                            cv2.putText(im0, str(classs)+" adjust", (10, 40),cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)

                            # print((len(try_icp_source) - len(temp_x)),buf_icp_target,buf_obj_xyxy)
                        elif (len(try_icp_source) - len(temp_x)) > 3 or (len(try_icp_source) - len(temp_x)) <0 :
                            buf_icp_target=[[0,0]]
                            buf_obj_xyxy = [objxmin,objymin,objxmax,objymax] 
                            # print((len(try_icp_source) - len(temp_x)),buf_icp_target,buf_obj_xyxy)
                            ww =1 
                            print('something go wrong')
                            print((len(try_icp_source) - len(temp_x)))

                        # else:
                        #     print((len(try_icp_source) - len(temp_x)))
                        #     print(buf_icp_target)
                        #     buf_obj_xyxy = [objxmin,objymin,objxmax,objymax] 
                        #     print("type error")
                        try_icp_target = buf_icp_target
                        obj_xyxy = buf_obj_xyxy
                        objxmin,objymin,objxmax,objmax = obj_xyxy
                        obj_w = (objxmax - objxmin)#物體的w
                        obj_h = (objymax - objymin)


                        # cv2.line(im0, y1p, y2p, (255,0,0), 2)#(0,255,0), 3) # g 
                        # cv2.line(im0, y1p, y3p, (255,0,0), 2)#(0,0,255), 3) # r
                        # cv2.line(im0, y4p, y2p, (255,0,0), 2)#(255,0,0), 3) # b 
                        # cv2.line(im0, y4p, y3p, (255,0,0), 2)#(255,255,255), 3) #white
                        # plot_one_box(obj_xyxy, im0, label=str(classs), color=[255,0,0], line_thickness=1)
                    # cv2.putText(im0, str(classs)+" order : "+str(lock_order), (10, 40),cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)

                    ############ icp      
                    ############ icp  
                    # for i in range(len(try_icp_target)):
                    #     cv2.circle(im0, [int(try_icp_target[i][0]*640),int(try_icp_target[i][1]*480)], 5, (0, 0, 255), -1) #
                    # for i in range(len(try_icp_source)):
                    #     cv2.circle(im0, [int(try_icp_source[i][0]*640),int(try_icp_source[i][1]*480)], 2, (0, 255, 0), -1) #
                    final_point  = []
                    if len(try_icp_target) != len(try_icp_source):
                        icp_mode='mine'
                    else:
                        icp_mode='mine' # 'open3d'   icp_kd'  'mine'
                    if icp_mode == 'open3d':
                        if len(try_icp_target) >0 :# try_icp_source try_icp_target
                            target = np.asarray(try_icp_target) # try_icp_target 
                            source = np.asarray(try_icp_source) #  用sop當要被改變得 #source 需要匹配的
                                                                #  辨識點為target 讓sop照著看到的方向轉變 
                            source_temp=np.asarray(source_temp)

                            newtarget = []
                            newsource = []
                            for i in range(len(try_icp_target)):
                                newtarget.append([target[i][0],target[i][1],0])
                                newtarget.append([target[i][0],target[i][1],1])
                            for i in range(len(try_icp_source)):
                                newsource.append([source[i][0],source[i][1],0])
                                newsource.append([source[i][0],source[i][1],1])

                            pcd_target = o3d.geometry.PointCloud()
                            pcd_source = o3d.geometry.PointCloud()
                            pcd_source_temp = o3d.geometry.PointCloud()
                            pcd_target.points = o3d.utility.Vector3dVector(newtarget)
                            pcd_source_temp.points = o3d.utility.Vector3dVector(newsource)
                            pcd_source_temp.points = o3d.utility.Vector3dVector(source_temp)
                            pcd_source.paint_uniform_color([1, 0, 0])    #source r色
                            pcd_target.paint_uniform_color([0, 0, 1])   #target b色
                            pcd_source_temp.paint_uniform_color([0, 1, 0]) #g
                            threshold = 0.1
                            trans_init = np.asarray([[1,0,0,0],   [0,1,0,0],   [0,0,1,0],   [0,0,0,1]]) # buf_transformation

                            reg_p2p = o3d.pipelines.registration.registration_icp(
                                pcd_source, pcd_target, threshold, trans_init,
                                o3d.pipelines.registration.TransformationEstimationPointToPoint(),o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 10000))
                            pcd_source_temp.transform(reg_p2p.transformation)
                            final_point = np.asarray(pcd_source_temp.points)
                    if icp_mode=='icp_kd':
                        if len(try_icp_target) >0 :# try_icp_source try_icp_target
                            reference_points = np.asarray(try_icp_source) # try_icp_target      # 需要被改變
                            points_to_be_aligned = np.asarray(try_icp_target)     # 被對齊 sop
                            # points_to_be_aligned += np.array([np.random.random_sample(), np.random.random_sample()])
                            transformation_history, final_point = icp(reference_points, points_to_be_aligned, verbose=True)
                            # print(final_point)
                    if icp_mode=='mine':
                        if len(try_icp_target) >0 :# try_icp_source try_icp_target
                            target = np.asarray(try_icp_target) # try_icp_target 
                            source = np.asarray(try_icp_source) #  用sop當要被改變得 #source 需要匹配的
                                                                #  辨識點為target 讓sop照著看到的方向轉變 

                            final_point,test_dis = DIY_ICP.Fit(source,target,show=0,show_f=0)
                            # final_point = np.asarray(pcd_source_temp.points)
                        # icp
                    # print(len(final_point),len(temp_sop_list))
                    if min(len(temp_sop_list),len(final_point)) > 0 :
                        for i in range( min(len(temp_sop_list),len(final_point)) ):
                            temp_sop_list[i][2] = final_point[i][0]
                            temp_sop_list[i][3] = final_point[i][1]
                            
                            cv2.circle(im0, [int(final_point[i][0]*(objxmax-objxmin)+objxmin),int(final_point[i][1]*(objymax-objymin)+objymin)], 3, (0, 255, 255), -1) #red 轉換後 lock_order 點
                    ############ icp       
                    ############ 判斷出type 後給順序 
                    # plot_one_box(obj_xyxy, im0, label=str(classs), color=[255,0,0], line_thickness=1)
                    if  ww ==0 :
                        for *xyxyo, conf, cls in reversed(det):
                            if int(cls) == 1:
                                xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                # cv2.circle(im0, [int(xywh[0]*img_x),int( xywh[1]*img_y)], 4, (255, 0, 0), -1) #green 辨識中心點在哪
                            if int(cls) == 0:
                                xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                # cv2.circle(im0, [int(xywh[0]*img_x),int( xywh[1]*img_y)], 4, (255, 0, 0), -1) #green 辨識中心點在哪
                                if  wrench_xyxy[2] > int(xywh[0]*img_x) > wrench_xyxy[0] and wrench_xyxy[3]>int(xywh[1]*img_y) > wrench_xyxy[1]: 
                                    pass #過濾qrcode 中可能出現洞
                                else:
                                    text = ""
                                    find_range = 1.5
                                    # print(lock_order,'lock_order')
                                    lock_order_in = int(lock_order)-1
                                    xmin = (temp_sop_list[lock_order_in][2]-temp_sop_list[lock_order_in][4]/find_range)*((objxmax-objxmin)*1.0) #(objxmax-objxmin)
                                    ymin = (temp_sop_list[lock_order_in][3]-temp_sop_list[lock_order_in][5]/find_range)*((objymax-objymin)*1.0) #(objymax-objymin)
                                    xmax = (temp_sop_list[lock_order_in][2]+temp_sop_list[lock_order_in][4]/find_range)*((objxmax-objxmin)*1.0) #(objxmax-objxmin)
                                    ymax = (temp_sop_list[lock_order_in][3]+temp_sop_list[lock_order_in][5]/find_range)*((objymax-objymin)*1.0) #(objymax-objymin)
                                    t = [xmin+objxmin,ymin+objymin,xmax+objxmin,ymax+objymin]
                                    r = [(temp_sop_list[lock_order_in][2]*(objxmax-objxmin)+objxmin),(temp_sop_list[lock_order_in][3]*(objymax-objymin)+objymin)]
                                    # cv2.circle(im0, [int(r[0]),int(r[1])], 3, (0, 0, 255), -1) #red 轉換後點
                                    # plot_one_box(t, im0, label=str(int(temp_sop_list[lock_order_in][0])), color=[0,0,255], line_thickness=1)#colors[int(cls)]

                                    if temp_sop_list[lock_order_in][6] == 0 :#and temp_sop_list[lock_order_in][0] == lock_order: # lock_order
                                        lock_order_xy = [(temp_sop_list[i][2]*(objxmax-objxmin)+objxmin),(temp_sop_list[i][3]*(objymax-objymin)+objymin)]
                                        # cv2.circle(im0, [int(lock_order_xy[0]),int(lock_order_xy[1])], 5, (0, 0, 255), -1) #red 轉換後 lock_order 點
                                        # block_order_xy= [(temp_sop_list[i][2]*(objxmax-objxmin)+objxmin),(temp_sop_list[i][3]*(objymax-objymin)+objymin)]
                                        # print(temp_sop_list[i][6],temp_sop_list[i][0])
                                        # plot_one_box(t, im0, label=str(int(temp_sop_list[i][0])), color=[0,0,255], line_thickness=1)#colors[int(cls)]
                                        if xmax+objxmin > xywh[0]*img_x > xmin+objxmin and ymax+objymin>xywh[1]*img_y > ymin+objymin: 
                                            # print([int(xywh[0]*img_x), int(xywh[1]*img_y)])
                                            block_order_xy= [int(xywh[0]*img_x), int(xywh[1]*img_y)]
                                            # print(block_order_xy)
                                            # lock_order_xy = block_order_xy
                                            cv2.putText(im0, str(int(temp_sop_list[lock_order_in][0])), (int(xywh[0]*img_x-5), int(xywh[1]*img_y)-30), cv2.FONT_HERSHEY_PLAIN, 4.0, (0, 0, 255), 2)
                                            cv2.circle(im0, [int(xywh[0]*img_x), int(xywh[1]*img_y)], 5, (0, 0, 255), -1) #red 轉換後 lock_order 點
                                    # else :
                                    #     if xmax+objxmin > xywh[0]*img_x > xmin+objxmin and ymax+objymin>xywh[1]*img_y > ymin+objymin: 
                                    #         cv2.putText(im0, str(int(temp_sop_list[lock_order_in][0]))+'_ ok ', (int(xywh[0]*img_x-5), int(xywh[1]*img_y)-30), cv2.FONT_HERSHEY_PLAIN, 4.0, (0, 0, 255), 2)
     
                    # print(block_order_xy,'b')
                    # print("_")
                    if block_order_xy != [0,0] and lock_order_xy ==[]:
                        lock_order_xy = [0,0]
                    elif block_order_xy != [0,0] and lock_order_xy !=[] and buf_icp_target != [[0,0,0]]:
                        lock_order_xy = block_order_xy
                    elif  block_order_xy == [0,0] and lock_order_xy !=[] :
                        lock_order_xy = lock_order_xy

                    if lock_order_xy != [0,0] and ww ==0 and temp_sop_list[lock_order_in][6] == 0:
                        if  wrench_xyxy[2] > int(lock_order_xy[0]) > wrench_xyxy[0] and wrench_xyxy[3]>int(lock_order_xy[1]) > wrench_xyxy[1]: 
                            key = cv2.waitKey(1)
                            mask = cv2.rectangle(blk, (int(wrench_xyxy[0]),int(wrench_xyxy[1])), (int(wrench_xyxy[2]),int(wrench_xyxy[3])),color=(0,255,0), thickness=-1 ) 
                            im0 = cv2.addWeighted(im0, 1, mask, 0.5, 0)
                            # 綠框可以所

                            if key == 13 and yolo_mode == 2: # enter 27 esc

                                print('鎖固完成')
                                temp_sop_list[lock_order_in][6] = 1 
                                lock_order_xy = []
                                
                                if int(lock_order) < len(temp_sop_list):
                                    try :
                                        path = './yoma_data'+"/order.txt"
                                        f = open(path, 'w')
                                        indata = str(int(lock_order)+1)

                                        print('indata',int(lock_order),len(temp_sop_list),indata)
                                        f.write(indata)
                                    except Exception as e:
                                        print(e)
                                    print("沒超過順序")
                                else :
                                    yolo_mode = 1

                                    print("都鎖完了!!!! 回辨識")
                                print(lock_order)
                        # elif lock_order_xy==[0,0]:
                        #     # print("有點在qrcode裡面")
                        #     pass
                        else :
                            # print("no")
                            mask = cv2.rectangle(blk, (int(wrench_xyxy[0]),int(wrench_xyxy[1])), (int(wrench_xyxy[2]),int(wrench_xyxy[3])),color=(0,0,255), thickness=-1 ) 
                            im0 = cv2.addWeighted(im0, 1, mask, 0.5, 0)                         
                if view_img:
                    # print("video")
                    cv2.imshow("yolov7", im0)#改視窗名字
                    key = cv2.waitKey(1)  # 1 millisecond
                    if key == 13 and yolo_mode == 1: # 13 enter 27 esc
                        lock_order = 0
                        if preds[0][maxlabel] >0.90:
                            yolo_mode = 2
                            print(preds[0][maxlabel])
                            print('into 鎖固')

                            path = './yoma_data'+"/order.txt"
                            try :
                                f = open(path, 'r')

                                lock_order = f.read()
                                print(lock_order)
                            except Exception as e:
                                print(e)
                            f.close()
                            
                        else :
                            print("辨識度太低 :　",preds[0][maxlabel])
                    elif yolo_mode == 2 and key ==27 :
                        yolo_mode = 1
                        print("classify")

                        try :
                            path = './yoma_data'+"/order.txt"
                            f = open(path, 'w')
                            indata = str(1)

                            f.write(indata)
                        except Exception as e:
                            print(e)
                        # lock_order = 0 # 初始化
                        # print(temp_sop_list)
                        print("classify")


                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0) #改這裡 可以改存的地方
                        print(f" The image with the result is saved in: {save_path}")
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)

                                w = 1280#int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = 720#int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                                # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
                            else:  # stream
                                fps, w, h = 30, 1280,720 #im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)

        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')
        # np.savetxt('./do aug type b.npy', np.asarray(result_list), delimiter=" ",fmt='%1.3f')#+"/1"+".txt"
        ########產生成功機率
        # result_len = len(result_list[0])
        # type_number = 1
        # print(result_len,result_list[0].count(type_number)/result_len,
        # result_list[1].count(type_number)/result_len,
        # result_list[2].count(type_number)/result_len)

        return  save_dir

    def hole_class_detect(save_img=False,opt = None): # 多了 model_weights_name let call class can chose .h5 file 
        source, weights, view_img, save_txt, imgsz, trace ,model_weights_name= opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace , opt.model_weights_name
        print(model_weights_name,imgsz)

        class_names,kmodel = load_test(model_weights_name) # load tf.keras.Model once !!

        RF=joblib.load('rf.model')
        xgboostModel = xgb.XGBClassifier()
        xgboostModel.load_model("xgb.json")
        print( class_names,kmodel)
        save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
        save_txt = 1
        yolo_mode = 1
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if trace:
            model = TracedModel(model, device, opt.img_size)
        if half:
            model.half()  # to FP16
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
        vid_path, vid_writer = None, None
        if webcam: #理論上進這裡 
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            cudnn.benchmark = True
            dataset = LoadImages(source, img_size=imgsz, stride=stride)
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1
        t0 = time.time()
        mid_preds = []
        buf_icp_target = []
        buf_obj_xyxy = []
        result_list = [[],[],[]]
        block_order_xy= [0,0]
        lock_order_xy = [0,0]
        #print("do once")
        try :
            path = './yoma_data'+"/order.txt"
            f = open(path, 'w')
            # indata = 1 # 初始直
            # print(indata)
            f.write(str(1))
        except Exception as e:
            print(e)
        for path, img, im0s, vid_cap in dataset:

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=opt.augment)[0]
            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=opt.augment)[0]
            t2 = time_synchronized()
            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t3 = time_synchronized()
            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                # txt_path = str(save_dir /  p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                obj_hole = []
                obj_hole2 = []

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                img_y,  img_x , _= im0.shape
                # print(img_y,  img_x)
                locktimes =1
                if yolo_mode == 1 :
                    temp_sop_list = []
                    temp_xmin=[]
                    temp_ymin=[]
                    temp_xmax=[]
                    temp_ymax=[]
                    temp_x = []
                    temp_y = []
                    for *xyxyo, conf, cls in reversed(det): #改成xyxyo 因為我要用xyxy
                        if int(cls) == 0:
                            xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh 影像的正規化
                            xmin = xywh[0]-xywh[2]/2
                            ymin = xywh[1]-xywh[3]/2
                            xmax = xywh[0]+xywh[2]/2
                            ymax = xywh[1]+xywh[3]/2
                            temp_xmin.append(xmin)
                            temp_ymin.append(ymin)
                            temp_xmax.append(xmax)
                            temp_ymax.append(ymax)
                            temp_x.append(xywh[0]*img_x)
                            temp_y.append(xywh[1]*img_y)


                        if int(cls) == 1:
                            xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh 影像的正規化
                            yolo_obj_xyxy = xyxy2xywh_transfer(xywh,[img_x,img_y],"xywh2xyxy") # yolo obj xmin ymin xmax ymax
                            # print(yolo_obj_xyxy)
                    obj_w,obj_h,objxmin,objymin,objxmax,objymax= get_obj_xyxy(temp_xmin,temp_ymin,temp_xmax,temp_ymax,temp_x,temp_y,im0.shape,mode = 1)
                    obj_xyxy=[objxmin,objymin,objxmax,objymax]
                    ####這邊只為了獲取obj大小 
                    try_icp_target = []
                    
                    ############classify type qq
                    for *xyxyo, conf, cls in reversed(det): #從640 480的正規化 轉成物體大小的正規化
                        if int(cls) == 0:

                            xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh 影像的正規化
                            xyxy = [(xywh[0]-xywh[2]/2)*img_x-objxmin ,(xywh[1]-xywh[3]/2)*img_y -objymin, (xywh[0]+xywh[2]/2)*img_x-objxmin , (xywh[1]+xywh[3]/2)*img_y-objymin]
                            # try_icp_target.append([xywh[0],xywh[1],0])
                            
                            # try_icp_target.append([round((xyxy[0]+xyxy[2])/2, 3),round((xyxy[1]+xyxy[3])/2, 3),0])
                            
                            xywh = xyxy2xywh_transfer(xyxy,[obj_w,obj_h],"xyxy2xywh")#物體的比例
                            xyxy = xyxy2xywh_transfer(xywh,[640,480],"xywh2xyxy")#給type的model轉回640*480 那時候model這樣train的
                                                                                 #可以看 point cloud yolo_桌 load_data 設定
                                                                                 #以固定畫面640*480沒差 但之後化面改變會出事  所以先處理
                            try_icp_target.append([xywh[0],xywh[1]])
                            obj_pc = ([round((xyxy[0]+xyxy[2])/2, 3),round((xyxy[1]+xyxy[3])/2, 3),0]) 
                            obj_pc2 = ([round((xyxy[0]+xyxy[2])/2, 3),round((xyxy[1]+xyxy[3])/2, 3)]) 
                            obj_hole2.append(obj_pc2)
                            obj_hole.append(obj_pc)
                        if int(cls) == 1:
                            xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh 影像的正規化 
                            # 可以抓obj 的參數 x center , ycenter ,w, h

                    obj_hole = obj_hole
                    obj_hole2 = obj_hole2

                    empty = [0,0,0]
                    if len(obj_hole) < 20  :
                        for b in range(20-len(obj_hole)):
                            obj_hole2.append([0,0])
                            obj_hole.append([b,b,b])
                    obj = (np.array(obj_hole)).reshape(1,20,3)
                    obj2 = (np.array(obj_hole2)).reshape(1,40)
                    preds = kmodel.predict(obj) # Point Net  
                    topreds = np.argmax(preds, -1) ## Point Net 分類用

                    result_list[0].append(topreds[0])
                    #list.count(obj)

                    mid_preds.append(topreds[0])
                    mid_times = 10
                    if len(mid_preds) >mid_times:
                        mid_preds.pop(0)
                    maxlabel = max(mid_preds,key=mid_preds.count)
                    classs = class_names[maxlabel]

                    # print("model_type : ",classification_type,classs)

                    cv2.putText(im0, str(classs), (40, 40), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                    sop_dir = './Eclatorq/sop/labels/'+str(classs)+".txt"
                    f = open(sop_dir)
                    try_icp_source = []
                    source_temp=[]

                    for line in f.readlines():
                        line = list(map(float, line.split(' ')))
                        temp_sop_list.append([line[0],line[1],line[2],line[3],line[4],line[5],0]) # 順序 , 種類 , x center ,y center , w , h 
                        try_icp_source.append([line[2],line[3]])
                        source_temp.append([line[2],line[3],0])

                    ############classify type 
                    ############ icp  
                    # for i in range(len(try_icp_target)):
                    #     cv2.circle(im0, [int(try_icp_target[i][0]*640),int(try_icp_target[i][1]*480)], 5, (0, 0, 255), -1) #
                    # for i in range(len(try_icp_source)):
                    #     cv2.circle(im0, [int(try_icp_source[i][0]*640),int(try_icp_source[i][1]*480)], 2, (0, 255, 0), -1) #

                    final_point =[]
                    if len(try_icp_target) >0 :# try_icp_source try_icp_target
                        target = np.asarray(try_icp_target) # try_icp_target 
                        source = np.asarray(try_icp_source) #  用sop當要被改變得 #source 需要匹配的
                                                            #  辨識點為target 讓sop照著看到的方向轉變 
                        # print('target',target) # ys yolo   >> sop 轉成yolo看到的
                        # print('source',source) # ys yolo   >> sop 轉成yolo看到的
                        final_point,test_dis= DIY_ICP.Fit(source,target,show=0,show_f=0)
                    if min(len(temp_sop_list),len(final_point)) > 0 :
                        for i in range( min(len(temp_sop_list),len(final_point)) ):
                            temp_sop_list[i][2] = final_point[i][0]
                            temp_sop_list[i][3] = final_point[i][1]
                            
                            cv2.circle(im0, [int(final_point[i][0]*(objxmax-objxmin)+objxmin),int(final_point[i][1]*(objymax-objymin)+objymin)], 3, (0, 255, 255), -1) #red 轉換後 lock_order 點
                    ############ icp       
                    ############ 判斷出type 後給順序 
                    # tttt_xyxy0=[]
                    # tttt_text = []
                    yyy=0
                    for *xyxyo, conf, cls in reversed(det):
                        if int(cls) == 0:
                            xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            cv2.circle(im0, [int(xywh[0]*img_x),int( xywh[1]*img_y)], 4, (255, 0, 0), -1) #green 辨識中心點在哪
                            text = ""
                            find_range = 1.2
                            for i in range(len(temp_sop_list)):
                                xmin = (temp_sop_list[i][2]-temp_sop_list[i][4]/find_range)*obj_w
                                ymin = (temp_sop_list[i][3]-temp_sop_list[i][5]/find_range)*obj_h
                                xmax = (temp_sop_list[i][2]+temp_sop_list[i][4]/find_range)*obj_w
                                ymax = (temp_sop_list[i][3]+temp_sop_list[i][5]/find_range)*obj_h
                                t = [xmin+objxmin,ymin+objymin,xmax+objxmin,ymax+objymin]
                                if xmax+objxmin > xywh[0]*img_x > xmin+objxmin and ymax+objymin>xywh[1]*img_y > ymin+objymin: 
                                    # tttt_xyxy0.append(xyxyo) #把洞跟順序分別抓下來  到時候用i就可以抓到對應的
                                    # tttt_text .append(str(text))
                                    plot_one_box(xyxyo, im0, label=str(int(temp_sop_list[i][0])), color=[0,0,255], line_thickness=3)#colors[int(cls)]
                                
                        if int(cls) == 1:
                            label = f'{names[int(cls)]} {conf:.2f}'    
                            plot_one_box(xyxyo, im0, label=label, color=colors[int(cls)], line_thickness=1)
                
                    buf_transformation = np.asarray([[1,0,0,0],   [0,1,0,0],   [0,0,1,0],   [0,0,0,1]]) 
                elif yolo_mode == 2 :#鎖固###########################################################################################################################################
                    wrench_tf = 0
                    path = './yoma_data'+"/order.txt"
                    try :
                        f = open(path, 'r')
                        lock_order = f.read()
                        # print(lock_order)
                    except Exception as e:
                        print(e)
                    f.close()
                    # lock_order_xy=[0,0]
                    #如果洞數量小於sop 2個 就用之前得物體大小    
                    sop_dir = './Eclatorq/sop/labels/'+str(classs)+".txt"
                    f = open(sop_dir)
                    # temp_sop_list = []
                    temp_xmin=[]
                    temp_ymin=[]
                    temp_xmax=[]
                    temp_ymax=[]
                    temp_x = []
                    temp_y = []
                    wrench_xyxy =[0,0,0,0]

                    for *xyxyo, conf, cls in reversed(det): #改成xyxyo 因為我要用xyxy
                        if int(cls) == 0:
                            xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh 影像的正規化
                            xmin = xywh[0]-xywh[2]/2
                            ymin = xywh[1]-xywh[3]/2
                            xmax = xywh[0]+xywh[2]/2
                            ymax = xywh[1]+xywh[3]/2
                            temp_xmin.append(xmin)
                            temp_ymin.append(ymin)
                            temp_xmax.append(xmax)
                            temp_ymax.append(ymax)
                            temp_x.append(xywh[0]*img_x)
                            temp_y.append(xywh[1]*img_y)
                        
                        if int(cls) == 1:
                            xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh 影像的正規化
                            yolo_obj_xyxy = xyxy2xywh_transfer(xywh,[img_x,img_y],"xywh2xyxy") # yolo obj xmin ymin xmax ymax
                            # print(yolo_obj_xyxy)
                        if int(cls) == 2: # 板手 wrench 
                            xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh 影像的正規化
                            yolo_wrench_xyxy = xyxy2xywh_transfer(xywh,[img_x,img_y],"xywh2xyxy") # yolo obj xmin ymin xmax ymax
                            # print(yolo_wrench_xyxy)
                            wrench_xyxy =[yolo_wrench_xyxy[0],yolo_wrench_xyxy[1],yolo_wrench_xyxy[2],yolo_wrench_xyxy[3]]
                            # print(wrench_xyxy)
                            plot_one_box(wrench_xyxy, im0, label="wrench", color=[255,255,255], line_thickness=1)
                            wrench_tf = 1 
                    obj_w,obj_h,objxmin,objymin,objxmax,objymax= get_obj_xyxy(temp_xmin,temp_ymin,temp_xmax,temp_ymax,temp_x,temp_y,im0.shape,mode = 1)
                    obj_xyxy=[objxmin,objymin,objxmax,objymax]

                    sop_dir = './Eclatorq/sop/labels/'+str(classs)+".txt"
                    f = open(sop_dir)
                    try_icp_source = []
                    for line in f.readlines():
                        #line = line.split(' ')
                        line = list(map(float, line.split(' ')))
                        sop_order = int(line[0])-1
                        temp_sop_list[sop_order][2] = line[2]
                        temp_sop_list[sop_order][3] = line[3] # temp_sop_list[order]
                        temp_sop_list[sop_order][4] = line[4]
                        temp_sop_list[sop_order][5] = line[5]
                        # temp_sop_list.append([line[0],line[1],line[2],line[3],line[4],line[5]]) # 順序 , 種類 , x center ,y center , w , h 
                        try_icp_source.append([line[2],line[3]]) # (x,y,z)
                        # try_icp_source.append([line[2]*640,line[3]*480,0]) # icp 畫面版 非正規畫板

                    qrcode_xyxy =[0,0,0,0]
                    qrcode_xy =[0,0]

                    blk = np.zeros(im0.shape, np.uint8)  


                    icp_target = []
                    temp_x = []
                    temp_y = []
                    for *xyxyo, conf, cls in reversed(det): #改成xyxyo 因為我要用xyxy
                        if int(cls) == 0:
                            
                            xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh 影像的正規化
                            temp_x.append(xywh[0]*img_x)
                            temp_y.append(xywh[1]*img_y)
                            xyxy = [(xywh[0]-xywh[2]/2)*img_x-objxmin ,(xywh[1]-xywh[3]/2)*img_y -objymin, (xywh[0]+xywh[2]/2)*img_x-objxmin , (xywh[1]+xywh[3]/2)*img_y-objymin]
                            if  wrench_xyxy[2] > int(xywh[0]*img_x) > wrench_xyxy[0]  and wrench_xyxy[3] >int(xywh[1]*img_y) > wrench_xyxy[1]: 
                                print("in wrench")
                                pass
                            else :
                                cv2.circle(im0, [int(xywh[0]*img_x),int( xywh[1]*img_y)], 5, (255, 0, 0), -1) #blue 辨識中心點在哪
                                xmin = xywh[0]-xywh[2]/2
                                ymin = xywh[1]-xywh[3]/2
                                xmax = xywh[0]+xywh[2]/2
                                ymax = xywh[1]+xywh[3]/2

                                xywh = xyxy2xywh_transfer(xyxy,[obj_w,obj_h],"xyxy2xywh")
                                xyxy = xyxy2xywh_transfer(xywh,[640,480],"xywh2xyxy")
                                xyxy_center = [xywh[0],xywh[1]] # xyxy_center

                                # xyxy_center = [(xyxy[0]+xyxy[2])/2,(xyxy[1]+xyxy[3])/2,0] # icp 畫面版 非正規畫板

                                icp_target.append(xyxy_center)
                            # print(xywh[0])
                        if int(cls) == 1:
                            label = f'{names[int(cls)]}'    
                            plot_one_box(xyxyo, im0, label=label, color=colors[int(cls)], line_thickness=1)
                    if wrench_tf ==1 :
                        #板手在畫面中 停止更新洞
                        if (len(try_icp_source) - len(temp_x)) ==0: 
                            # print(y1p,y2p,y3p,y4p)
                            buf_obj_xyxy = [objxmin,objymin,objxmax,objymax] # 有這個代表正常
                            # print(buf_obj_xyxy)
                            # buf_icp_target = icp_target
                            # print("1")
                            cv2.putText(im0, str(classs)+" order : "+str(lock_order), (10, 40),cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                            ww=0
                            # print((len(try_icp_source) - len(temp_x)),buf_icp_target,buf_obj_xyxy)

                    if wrench_tf ==0 :
                        if (len(try_icp_source) - len(temp_x)) ==0: 
                            # print(y1p,y2p,y3p,y4p)
                            buf_obj_xyxy = [objxmin,objymin,objxmax,objymax] # 有這個代表正常
                            # print(buf_obj_xyxy)
                            buf_icp_target = icp_target
                            # print("1")
                            cv2.putText(im0, str(classs)+" order : "+str(lock_order), (10, 40),cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                            ww=0
                            # print((len(try_icp_source) - len(temp_x)),buf_icp_target,buf_obj_xyxy)
                        elif (len(try_icp_source) - len(temp_x)) >= 1 :#被擋住1洞 and buf_icp_target == [] and buf_icp_target != [[0,0,0]]
                            # buf_icp_target=[[0,0,0]]
                            buf_icp_target = icp_target

                            buf_obj_xyxy = [objxmin,objymin,objxmax,objymax] # 有這個代表正常
                            # print((len(try_icp_source) - len(temp_x)),buf_icp_target,buf_obj_xyxy)
                            cv2.putText(im0, str(classs)+" order : "+str(lock_order), (10, 40),cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                            ww=0
                            print(buf_obj_xyxy)
                        elif (len(try_icp_source) - len(temp_x)) >= 3 :#被擋住2洞 and buf_icp_target == [] and buf_icp_target != [[0,0,0]]
                            # buf_icp_target=[[0,0,0]]

                            buf_obj_xyxy = [objxmin,objymin,objxmax,objymax] 
                            print("請調整")
                            ww =1  #目前+ww有比較不會一直按enter 一直鎖固 但應該需要rc綠波 不然畫面一直跳
                            #目前
                            cv2.putText(im0, str(classs)+" adjust", (10, 40),cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)

                            # print((len(try_icp_source) - len(temp_x)),buf_icp_target,buf_obj_xyxy)
                        elif (len(try_icp_source) - len(temp_x)) > 3 or (len(try_icp_source) - len(temp_x)) <0 :
                            buf_icp_target=[[0,0]]
                            buf_obj_xyxy = [objxmin,objymin,objxmax,objymax] 
                            # print((len(try_icp_source) - len(temp_x)),buf_icp_target,buf_obj_xyxy)
                            ww =1 
                            print('something go wrong')
                            print((len(try_icp_source) - len(temp_x)))

                    # if len(temp_x) > 0 :  # 可以把xmin 改掉
                    #     # print((len(try_icp_source) - len(temp_x)))

                        
                    #     elif (len(try_icp_source) - len(temp_x)) >= 3 :#被擋住2洞 and buf_icp_target == [] and buf_icp_target != [[0,0,0]]
                    #         # buf_icp_target=[[0,0,0]]

                    #         buf_obj_xyxy = [objxmin,objymin,objxmax,objymax] 
                    #         print("請調整")
                    #         ww =1  #目前+ww有比較不會一直按enter 一直鎖固 但應該需要rc綠波 不然畫面一直跳
                    #         #目前
                    #         cv2.putText(im0, str(classs)+" adjust", (10, 40),cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)

                    #         # print((len(try_icp_source) - len(temp_x)),buf_icp_target,buf_obj_xyxy)
                    #     elif (len(try_icp_source) - len(temp_x)) > 3 or (len(try_icp_source) - len(temp_x)) <0 :
                    #         buf_icp_target=[[0,0]]
                    #         buf_obj_xyxy = [objxmin,objymin,objxmax,objymax] 
                    #         # print((len(try_icp_source) - len(temp_x)),buf_icp_target,buf_obj_xyxy)
                    #         ww =1 
                    #         print('something go wrong')
                    #         print((len(try_icp_source) - len(temp_x)))

                        # else:
                        #     print((len(try_icp_source) - len(temp_x)))
                        #     print(buf_icp_target)
                        #     buf_obj_xyxy = [objxmin,objymin,objxmax,objymax] 
                        #     print("type error")
                        try_icp_target = buf_icp_target
                        obj_xyxy = buf_obj_xyxy
                        objxmin,objymin,objxmax,objmax = obj_xyxy
                        obj_w = (objxmax - objxmin)#物體的w
                        obj_h = (objymax - objymin)


                    final_point  = []
                   
                    if len(try_icp_target) >0 :# try_icp_source try_icp_target
                        target = np.asarray(try_icp_target) # try_icp_target 
                        source = np.asarray(try_icp_source) #  用sop當要被改變得 #source 需要匹配的
                                                            #  辨識點為target 讓sop照著看到的方向轉變 

                        final_point,test_dis = DIY_ICP.Fit(source,target,show=0,show_f=0)
                            # final_point = np.asarray(pcd_source_temp.points)
                        # icp
                    # print(len(final_point),len(temp_sop_list))
                    if min(len(temp_sop_list),len(final_point)) > 0 :
                        for i in range( min(len(temp_sop_list),len(final_point)) ):
                            temp_sop_list[i][2] = final_point[i][0]
                            temp_sop_list[i][3] = final_point[i][1]
                            
                            cv2.circle(im0, [int(final_point[i][0]*(objxmax-objxmin)+objxmin),int(final_point[i][1]*(objymax-objymin)+objymin)], 3, (0, 255, 255), -1) #red 轉換後 lock_order 點
                    ############ icp       
                    ############ 判斷出type 後給順序 
                    # plot_one_box(obj_xyxy, im0, label=str(classs), color=[255,0,0], line_thickness=1)
                    if  ww ==0 :
                        for *xyxyo, conf, cls in reversed(det):
                            if int(cls) == 1:
                                xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                # cv2.circle(im0, [int(xywh[0]*img_x),int( xywh[1]*img_y)], 4, (255, 0, 0), -1) #green 辨識中心點在哪
                            if int(cls) == 0:
                                xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                # cv2.circle(im0, [int(xywh[0]*img_x),int( xywh[1]*img_y)], 4, (255, 0, 0), -1) #green 辨識中心點在哪
                                if  wrench_xyxy[2] > int(xywh[0]*img_x) > wrench_xyxy[0] and wrench_xyxy[3]>int(xywh[1]*img_y) > wrench_xyxy[1]: 
                                    pass #過濾qrcode 中可能出現洞
                                else:
                                    text = ""
                                    find_range = 1.5
                                    # print(lock_order,'lock_order')
                                    lock_order_in = int(lock_order)-1
                                    xmin = (temp_sop_list[lock_order_in][2]-temp_sop_list[lock_order_in][4]/find_range)*((objxmax-objxmin)*1.0) #(objxmax-objxmin)
                                    ymin = (temp_sop_list[lock_order_in][3]-temp_sop_list[lock_order_in][5]/find_range)*((objymax-objymin)*1.0) #(objymax-objymin)
                                    xmax = (temp_sop_list[lock_order_in][2]+temp_sop_list[lock_order_in][4]/find_range)*((objxmax-objxmin)*1.0) #(objxmax-objxmin)
                                    ymax = (temp_sop_list[lock_order_in][3]+temp_sop_list[lock_order_in][5]/find_range)*((objymax-objymin)*1.0) #(objymax-objymin)
                                    t = [xmin+objxmin,ymin+objymin,xmax+objxmin,ymax+objymin]
                                    r = [(temp_sop_list[lock_order_in][2]*(objxmax-objxmin)+objxmin),(temp_sop_list[lock_order_in][3]*(objymax-objymin)+objymin)]
                                    # cv2.circle(im0, [int(r[0]),int(r[1])], 3, (0, 0, 255), -1) #red 轉換後點
                                    # plot_one_box(t, im0, label=str(int(temp_sop_list[lock_order_in][0])), color=[0,0,255], line_thickness=1)#colors[int(cls)]

                                    if temp_sop_list[lock_order_in][6] == 0 :#and temp_sop_list[lock_order_in][0] == lock_order: # lock_order
                                        lock_order_xy = [(temp_sop_list[i][2]*(objxmax-objxmin)+objxmin),(temp_sop_list[i][3]*(objymax-objymin)+objymin)]
                                        # cv2.circle(im0, [int(lock_order_xy[0]),int(lock_order_xy[1])], 5, (0, 0, 255), -1) #red 轉換後 lock_order 點
                                        # block_order_xy= [(temp_sop_list[i][2]*(objxmax-objxmin)+objxmin),(temp_sop_list[i][3]*(objymax-objymin)+objymin)]
                                        # print(temp_sop_list[i][6],temp_sop_list[i][0])
                                        # plot_one_box(t, im0, label=str(int(temp_sop_list[i][0])), color=[0,0,255], line_thickness=1)#colors[int(cls)]
                                        if xmax+objxmin > xywh[0]*img_x > xmin+objxmin and ymax+objymin>xywh[1]*img_y > ymin+objymin: 
                                            # print([int(xywh[0]*img_x), int(xywh[1]*img_y)])
                                            block_order_xy= [int(xywh[0]*img_x), int(xywh[1]*img_y)]
                                            # print(block_order_xy)
                                            # lock_order_xy = block_order_xy
                                            cv2.putText(im0, str(int(temp_sop_list[lock_order_in][0])), (int(xywh[0]*img_x-5), int(xywh[1]*img_y)-30), cv2.FONT_HERSHEY_PLAIN, 4.0, (0, 0, 255), 2)
                                            cv2.circle(im0, [int(xywh[0]*img_x), int(xywh[1]*img_y)], 5, (0, 0, 255), -1) #red 轉換後 lock_order 點
                                    # else :
                                    #     if xmax+objxmin > xywh[0]*img_x > xmin+objxmin and ymax+objymin>xywh[1]*img_y > ymin+objymin: 
                                    #         cv2.putText(im0, str(int(temp_sop_list[lock_order_in][0]))+'_ ok ', (int(xywh[0]*img_x-5), int(xywh[1]*img_y)-30), cv2.FONT_HERSHEY_PLAIN, 4.0, (0, 0, 255), 2)
     
                    # print(block_order_xy,'b')
                    # print("_")
                    if block_order_xy != [0,0] and lock_order_xy ==[]:
                        lock_order_xy = [0,0]
                    elif block_order_xy != [0,0] and lock_order_xy !=[] and buf_icp_target != [[0,0,0]]:
                        lock_order_xy = block_order_xy
                    elif  block_order_xy == [0,0] and lock_order_xy !=[] :
                        lock_order_xy = lock_order_xy

                    if lock_order_xy != [0,0] and ww ==0 and temp_sop_list[lock_order_in][6] == 0:
                        if  wrench_xyxy[2] > int(lock_order_xy[0]) > wrench_xyxy[0] and wrench_xyxy[3]>int(lock_order_xy[1]) > wrench_xyxy[1]: 
                            key = cv2.waitKey(1)
                            mask = cv2.rectangle(blk, (int(wrench_xyxy[0]),int(wrench_xyxy[1])), (int(wrench_xyxy[2]),int(wrench_xyxy[3])),color=(0,255,0), thickness=-1 ) 
                            im0 = cv2.addWeighted(im0, 1, mask, 0.5, 0)
                            # 綠框可以所

                            if key == 13 and yolo_mode == 2: # enter 27 esc

                                print('鎖固完成')
                                temp_sop_list[lock_order_in][6] = 1 
                                lock_order_xy = []
                                
                                if int(lock_order) < len(temp_sop_list):
                                    try :
                                        path = './yoma_data'+"/order.txt"
                                        f = open(path, 'w')
                                        indata = str(int(lock_order)+1)

                                        print('indata',int(lock_order),len(temp_sop_list),indata)
                                        f.write(indata)
                                    except Exception as e:
                                        print(e)
                                    print("沒超過順序")
                                else :
                                    yolo_mode = 1

                                    print("都鎖完了!!!! 回辨識")
                                    try :
                                        path = './yoma_data'+"/order.txt"
                                        f = open(path, 'w')
                                        indata = str(1)

                                        f.write(indata)
                                    except Exception as e:
                                        print(e)
                                print(lock_order)
                        # elif lock_order_xy==[0,0]:
                        #     # print("有點在qrcode裡面")
                        #     pass
                        else :
                            # print("no")
                            mask = cv2.rectangle(blk, (int(wrench_xyxy[0]),int(wrench_xyxy[1])), (int(wrench_xyxy[2]),int(wrench_xyxy[3])),color=(0,0,255), thickness=-1 ) 
                            im0 = cv2.addWeighted(im0, 1, mask, 0.5, 0)                         
                if view_img:
                    # print("video")
                    cv2.imshow("yolov7", im0)#改視窗名字
                    key = cv2.waitKey(1)  # 1 millisecond
                    if key == 13 and yolo_mode == 1: # 13 enter 27 esc
                        lock_order = 0
                        if preds[0][maxlabel] >0.90:
                            yolo_mode = 2
                            print(preds[0][maxlabel])
                            print('into 鎖固')

                            path = './yoma_data'+"/order.txt"
                            try :
                                f = open(path, 'r')

                                lock_order = f.read()
                                print(lock_order)
                            except Exception as e:
                                print(e)
                            f.close()
                            
                        else :
                            print("辨識度太低 :　",preds[0][maxlabel])
                    elif yolo_mode == 2 and key ==27 :
                        yolo_mode = 1
                        print("classify")

                        try :
                            path = './yoma_data'+"/order.txt"
                            f = open(path, 'w')
                            indata = str(1)

                            f.write(indata)
                        except Exception as e:
                            print(e)
                        # lock_order = 0 # 初始化
                        # print(temp_sop_list)
                        print("classify")


                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0) #改這裡 可以改存的地方
                        print(f" The image with the result is saved in: {save_path}")
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)

                                w = 1280#int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = 720#int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                                # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
                            else:  # stream
                                fps, w, h = 30, 1280,720 #im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)

        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')
        # np.savetxt('./do aug type b.npy', np.asarray(result_list), delimiter=" ",fmt='%1.3f')#+"/1"+".txt"
        ########產生成功機率
        # result_len = len(result_list[0])
        # type_number = 1
        # print(result_len,result_list[0].count(type_number)/result_len,
        # result_list[1].count(type_number)/result_len,
        # result_list[2].count(type_number)/result_len)

        return  save_dir

# filename = os.path.basename(__file__)
# filename=filename[17:len(filename)-3]
# print("filename :",filename)

NUM_POINTS = 2048
NUM_CLASSES = 10
history_list=[]
model_list=[]
preds_list=[]
class_names_list= []
label_list=[]
color = [[0.5,0,0],'b','g','r','y','k','c','m']
def xyxy2xywh_transfer(temp_xy,imgyx,mode):
    # print(imgyx)
    if len(temp_xy) >0 :
        # if imgyx[1] > imgyx[0]:
        #     print(imgyx)
        #     print("xyxy2xywh_transfer img 垂直")# y = h > x = w  ----  h > w 如果不是垂直  imgyx 相反
        # elif imgyx[0] > imgyx[1]:
            # print("xyxy2xywh_transfer img 水平 正常") #
        if mode == 'xyxy2xywh':
            xywh = [((temp_xy[0]+temp_xy[2])/2)/imgyx[0],((temp_xy[1]+temp_xy[3])/2)/imgyx[1],abs((temp_xy[0]-temp_xy[2])) /imgyx[0],abs((temp_xy[1]-temp_xy[3])) /imgyx[1]]
            return xywh
        elif mode == 'xywh2xyxy':
            xyxy = [(temp_xy[0]-temp_xy[2]/2)*imgyx[0] ,(temp_xy[1]-temp_xy[3]/2)*imgyx[1] , (temp_xy[0]+temp_xy[2]/2)*imgyx[0] , (temp_xy[1]+temp_xy[3]/2)*imgyx[1]]
            return xyxy





def classify2tensor ():

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yoma_data/weights/best.pt', help='model.pt path(s)') # 辨識洞 基本上固定
    parser.add_argument('--model_weights_name', action='store_true',default='keras.h5', help='model.h5') # 辨識種類 名稱固定 但可能可以改
    parser.add_argument('--source', type=str, default=str(1), help='source')  # file/folder  inference/images  , 0 for webcam
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
    # print(opt.source)

    # yc.hole_detect(opt)
    yc.product_testing_data(opt)


#fit_list = [x_train,x_train]


if __name__ == "__main__":
    yc = classify()
    classify2tensor()
# for i in range (model_num): # 
#     history = "history"+str(i)
#     model_save = "model"+str(i)
#     print(model_save)
#     model_list.append(model_save) 
#     model = eval("model"+str(i))
#     model_weights_name= filename+'_yolo_'+str(model_save)+'.h5'
#     #print(model_list)
#     class_names,train_points,train_labels = load_dataset(DATA_DIR[i])
#     x_train,x_test,y_train,y_test = Kfold_ramdom(train_points,train_labels,splits,666,87)
#     print(class_names)
#     if fit_tf == True :
#         history,preds,label= model(x_train,y_train,x_test,y_test,class_names,BATCH_SIZE_lsit[i],epochs,model_weights_name,fit_tf) #BATCH_SIZE,epochs
#         history_list.append(history)

#     elif fit_tf == False:
#         preds,label= model(x_train,y_train,x_test,y_test,class_names,BATCH_SIZE_lsit[i],epochs,model_weights_name,fit_tf) #BATCH_SIZE,epochs

#     buf_pr = preds##給後面顯示用
#     preds = np.argmax(preds, -1) ## 分類
#     label_list.append(label)
#     preds_list.append(preds)
#     class_names_list.append(class_names)
#     print(class_names_list)

# plot_confusion_matrix(label_list, preds_list, classes=class_names_list, normalize=False,title=model_list)
# try :
#     if fit_tf == True:
#         plot_history(history_list)
#         print("test")
# except ValueError:
#         pass
# plt.show()
# [[    0.94431   ,   0.3401], [    0.78436   ,  0.89213], [    0.07109   ,  0.57487], [    0.77133   ,  0.42386], [    0.36374   ,  0.26015], [    0.69431   ,  0.66117], [    0.70498   , 0.093909], [    0.27607   ,  0.47716], [    0.48341   ,  0.68655], [    0.63389   ,  0.26523], [    0.29147   , 0.074873], [    0.39336   ,  0.90736]]
 # [[    0.92824   ,  0.63092],[    0.59838   ,  0.68579], [    0.65625   ,  0.28803], [    0.39352   , 0.077307], [     0.7338   ,  0.10349], [    0.73727   ,  0.51746], [    0.45602   ,  0.25686], [   0.068287   ,  0.33541], [    0.24769   ,  0.43516], [    0.31019   ,  0.67082], [    0.21296   ,  0.89152], [    0.66898  ,   0.91272]]


