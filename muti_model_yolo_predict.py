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
qreader_reader = QReader()
# source, weights, view_img, save_txt, imgsz, trace ,classification_type= opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace , opt.classification_type
# print(source, weights, view_img, save_txt, imgsz, trace)
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

DATA_DIR = "C:/Users/karta/OneDrive/桌面/one_pycharm/yolo"  # model 0 用10種 model 1 用5種
classification_type = "PN" # XGB # RF

class_names,train_points,train_labels = load_dataset(DATA_DIR)
if classification_type =="RF" :#Random Forest
    import joblib
    RF=joblib.load('rf.model')
elif classification_type =="XGB": 
    import xgboost as xgb
    xgboostModel = xgb.XGBClassifier()
    xgboostModel.load_model("xgb.json")
elif classification_type =="PN": # Point Net
    model_weights_name= '20230918'+'.h5' #
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
    # model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    kmodel.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["sparse_categorical_accuracy"], )
    #model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=1)
    kmodel.load_weights(model_weights_name)

class classify():
    def detect(save_img=False,opt = None):
        source, weights, view_img, save_txt, imgsz, trace ,classification_type= opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace , opt.classification_type
        print(source, weights, view_img, save_txt, imgsz, trace)
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
        if webcam:
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

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                img_y,  img_x , _= im0.shape
                if yolo_mode == 1 :
                    temp_sop_list = []
                    temp_xmin=[]
                    temp_ymin=[]
                    temp_xmax=[]
                    temp_ymax=[]
                    for *xyxyo, conf, cls in reversed(det): #改成xyxyo 因為我要用xyxy
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:# with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        xmin = xywh[0]-xywh[2]/2
                        ymin = xywh[1]-xywh[3]/2
                        xmax = xywh[0]+xywh[2]/2
                        ymax = xywh[1]+xywh[3]/2
                        temp_xmin.append(xmin)
                        temp_ymin.append(ymin)
                        temp_xmax.append(xmax)
                        temp_ymax.append(ymax)
                        # plot_one_box_notext(xyxyo, im0, label="", color=colors[int(cls)], line_thickness=3)
                    if len(temp_xmin) > 0 :
                        objxmin = min(temp_xmin)*img_x
                        objymin = min(temp_ymin)*img_y
                        objxmax = max(temp_xmax)*img_x
                        objymax = max(temp_ymax)*img_y
                        tttt = [objxmin,objymin,objxmax,objymax] 
                        nw = (objxmax - objxmin)#物體的w
                        nh = (objymax - objymin)
                        # print(tttt)
                        plot_one_box(tttt, im0, label="obj", color=[255,0,0], line_thickness=1)
                    ####這邊只為了獲取obj大小 
                    try_icp_source = []
                    
                    ############classify type 
                    for *xyxyo, conf, cls in reversed(det): #從640 480的正規化 轉成物體大小的正規化
                        xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh 影像的正規化
                        xyxy = [(xywh[0]-xywh[2]/2)*img_x-objxmin ,(xywh[1]-xywh[3]/2)*img_y -objymin, (xywh[0]+xywh[2]/2)*img_x-objxmin , (xywh[1]+xywh[3]/2)*img_y-objymin]
                        
                        xywh = xyxy2xywh_transfer(xyxy,[nw,nh],"xyxy2xywh")#物體的比例
                        xyxy = xyxy2xywh_transfer(xywh,[640,480],"xywh2xyxy")#給type的model轉回640*480 那時候model這樣train的
                                                                             #可以看 point cloud yolo_桌 load_data 設定
                                                                             #以固定畫面640*480沒差 但之後化面改變會出事  所以先處理
                        try_icp_source.append([xywh[0],xywh[1],0])
                        if classification_type =="RF" or classification_type =="XGB" :  #Random Forest
                            obj_pc = ([round((xyxy[0]+xyxy[2])/2, 3),round((xyxy[1]+xyxy[3])/2, 3)]) 
                        elif classification_type =="PN": # Point Net
                            obj_pc = ([round((xyxy[0]+xyxy[2])/2, 3),round((xyxy[1]+xyxy[3])/2, 3),0]) 
                        # obj_pc = ([round((xyxy[0]+xyxy[2])/2, 3),round((xyxy[1]+xyxy[3])/2, 3)]) 
                        #obj_pc = ([round((xyxy[0]+xyxy[2])/2, 3),round((xyxy[1]+xyxy[3])/2, 3),0]) #point net
                        obj_hole.append(obj_pc)

                    obj_hole = obj_hole
                    if classification_type =="RF" or classification_type =="XGB":
                        empty = [0,0]
                    elif classification_type =="PN":
                        empty = [0,0,0]
                    if len(obj_hole) < 20  :
                        add_num = 20-len(obj_hole)
                        list(obj_hole)
                        for j in range(add_num):
                            obj_hole.append(empty) # 代表填充的
                            #obj_hole.append([0,0,0]) # 代表填充的point net

                    # obj = (np.array(obj_hole)).reshape(1,20,3)
                    if classification_type =="RF" :#Random Forest
                        obj = (np.array(obj_hole)).reshape(1,40)
                        topreds = RF.predict(obj) 
                    elif classification_type =="XGB": 
                        obj = (np.array(obj_hole)).reshape(1,40)
                        topreds = xgboostModel.predict(obj) # Point Net  
                    elif classification_type =="PN": # Point Net
                        obj = (np.array(obj_hole)).reshape(1,20,3)
                        preds = kmodel.predict(obj) # Point Net  
                        topreds = np.argmax(preds, -1) ## Point Net 分類用

                    # classs = class_names[topreds][0]
                    # 如果type model不會出錯 可以拿掉下面的 用上面那句
                    # 但保留可以增加穩定性

                    print("model_type : ",classification_type,topreds[0])

                    mid_preds.append(topreds[0])
                    mid_times = 10
                    if len(mid_preds) >mid_times:
                        mid_preds.pop(0)
                    maxlabel = max(mid_preds,key=mid_preds.count)
                    classs = class_names[maxlabel]

                    cv2.putText(im0, str(classs), (40, 40), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                    sop_dir = './Eclatorq/sop/labels/'+str(classs)+".txt"
                    f = open(sop_dir)
                    try_icp_target = []
                    for line in f.readlines():
                        #line = line.split(' ')
                        line = list(map(float, line.split(' ')))
                        temp_sop_list.append([line[0],line[1],line[2],line[3],line[4],line[5],0]) # 順序 , 種類 , x center ,y center , w , h 
                        try_icp_target.append([line[2],line[3],0])
                        #load sop進來 下面比較 #######這邊要去改 抓物體後得比例 現在是影像的比例
                    ############classify type 
                    ############ icp 要確認     
                    ############ icp 要確認     

                    #source >> type
                    import open3d as o3d
                    import copy
                    def draw_registration_result(source, target, transformation):
                        source_temp = copy.deepcopy(source)
                        target_temp = copy.deepcopy(target)
                        source_temp.paint_uniform_color([1, 0.706, 0])
                        target_temp.paint_uniform_color([0, 0.651, 0.929]) # 蓝色
                        source_temp.transform(transformation)
                        # source.paint_uniform_color([0, 0.651, 0.929])
                        # o3d.visualization.draw_geometries([source])
                        # o3d.visualization.draw_geometries([source_temp, target_temp])
                        # o3d.visualization.draw_geometries([source,source_temp])

                        o3d.visualization.draw_geometries([source_temp, target_temp])

                    if len(try_icp_source) >0 :# try_icp_source try_icp_target
                        target = np.asarray(try_icp_source) # 
                        source = np.asarray(try_icp_target) #  用sop當要被改變得source
                                                            #  辨識點為target 讓sop照著看到的方向轉變 
                        pcd_target = o3d.geometry.PointCloud()
                        pcd_source = o3d.geometry.PointCloud()
                        source_temp = o3d.geometry.PointCloud()

                        pcd_target.points = o3d.utility.Vector3dVector(target)
                        pcd_source.points = o3d.utility.Vector3dVector(source)

                        pcd_source.paint_uniform_color([1, 0, 0])    #source r色
                        pcd_target.paint_uniform_color([0, 0, 1])#target b色
                        source_temp.paint_uniform_color([0, 1, 0]) #g

                        threshold = 0.3
                        trans_init = np.asarray([[1,0,0,0],   # 4x4 identity matrix，这是一个转换矩阵，
                                                 [0,1,0,0],   # 象征着没有任何位移，没有任何旋转，我们输入
                                                 [0,0,1,0],   # 这个矩阵为初始变换
                                                 [0,0,0,1]])

                        # print("Apply point-to-point ICP")
                        reg_p2p = o3d.pipelines.registration.registration_icp(
                            pcd_source, pcd_target, threshold, trans_init,
                            o3d.pipelines.registration.TransformationEstimationPointToPoint(),o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 2000))
                        # print(reg_p2p)

                        source_temp = copy.deepcopy(pcd_source)
                        source_temp.transform(reg_p2p.transformation)
                        final_point = np.asarray(source_temp.points)

                        # print(np.asarray(pcd_source.points))
                        # print("Transformation : \n",final_point)
                        # print("Transformation is:")
                        # print(reg_p2p.transformation)
                        ####### plot 部分 
                        # o3d.visualization.draw_geometries([source_temp, pcd_source,pcd_target])
                        # draw_registration_result(pcd_source, pcd_target, reg_p2p.transformation)

                        for i in range(len(temp_sop_list)):
                            temp_sop_list[i][2]
                            source = [[temp_sop_list[i][2],temp_sop_list[i][3],0]]
                            

                            pcd_source = o3d.geometry.PointCloud()
                            pcd_source.points = o3d.utility.Vector3dVector(source)
                            source_temp = copy.deepcopy(pcd_source)
                            source_temp.paint_uniform_color([0, 1, 0])
                            source_temp.transform(reg_p2p.transformation)
                            final_point = np.asarray(source_temp.points)
                            # print("原本 : ",source) 
                            # print("轉換 : ",final_point)
                            temp_sop_list[i][2] =final_point[0][0]
                            temp_sop_list[i][3] =final_point[0][1]

                    ############ icp 要確認     
                    ############ icp 要確認      

                    ############ 判斷出type 後給順序 
                    tttt_xyxy0=[]
                    tttt_text = []
                    for *xyxyo, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        cv2.circle(im0, [int(xywh[0]*img_x),int( xywh[1]*img_y)], 5, (0, 255, 0), -1) #green 辨識中心點在哪

                        text = ""
                        find_range = 1.5
                        for i in range(len(temp_sop_list)):
                            xmin = (temp_sop_list[i][2]-temp_sop_list[i][4]/find_range)*(objxmax-objxmin)
                            ymin = (temp_sop_list[i][3]-temp_sop_list[i][5]/find_range)*(objymax-objymin)
                            xmax = (temp_sop_list[i][2]+temp_sop_list[i][4]/find_range)*(objxmax-objxmin)
                            ymax = (temp_sop_list[i][3]+temp_sop_list[i][5]/find_range)*(objymax-objymin)
                            t = [xmin+objxmin,ymin+objymin,xmax+objxmin,ymax+objymin]
                            # plot_one_box(t, im0, label="", color=[255,0,0], line_thickness=1) # 載入sop預估圖
                            # 下面會需要加 objxmin objymin 因為sop load 進來會是以0,0為原點
                            # 但這邊需要以obj最小值為原點
                            # 這邊可以去看gui_control 那邊的 save()的設定
                            # cv2.circle(im0, [int((t[0]+t[2])/2),int((t[1]+t[3])/2)], 5, (255, 0, 0), -1) #blue sop中心點 

                            if xmax+objxmin > xywh[0]*img_x > xmin+objxmin and ymax+objymin>xywh[1]*img_y > ymin+objymin: 
                                # cv2.circle(im0, [int((t[0]+t[2])/2),int((t[1]+t[3])/2)], 5, (255, 0, 0), -1) #blue sop中心點 
                                text = int(temp_sop_list[i][0])
                                label = str(text)
                                tttt_xyxy0.append(xyxyo)
                                tttt_text .append(str(text))
                                plot_one_box(xyxyo, im0, label=label, color=[0,0,255], line_thickness=3)#colors[int(cls)]
                    ############ 判斷出type 後給順序
                elif yolo_mode == 2 :
                    #如果洞數量小於sop 2個 就用之前得物體大小    
                    sop_dir = './Eclatorq/sop/labels/'+str(classs)+".txt"
                    f = open(sop_dir)
                    try_icp_source = []
                    for line in f.readlines():
                        #line = line.split(' ')
                        line = list(map(float, line.split(' ')))
                        zz = int(line[0])-1
                        temp_sop_list[zz][2] = line[2]
                        temp_sop_list[zz][3] = line[3]
                        temp_sop_list[zz][4] = line[4]
                        temp_sop_list[zz][5] = line[5]
                        # temp_sop_list.append([line[0],line[1],line[2],line[3],line[4],line[5]]) # 順序 , 種類 , x center ,y center , w , h 
                        try_icp_source.append([line[2],line[3],0])
                        # try_icp_source.append([line[2]*640,line[3]*480,0])


                    qrcode_xyxy =[0,0,0,0]
                    rr=[0,0]
                    blk = np.zeros(im0.shape, np.uint8)  
                    decoded_text = qreader_reader.detect(image=im0)
                    if decoded_text != []:
                        # key = cv2.waitKey(1)
                        qrcode_xyxy = decoded_text[0]['bbox_xyxy']
                        qr_center = [(qrcode_xyxy[0]+qrcode_xyxy[2])/2,(qrcode_xyxy[1]+qrcode_xyxy[3])/2]
                        # cv2.circle(im0, [int(qr_center[0]),int( qr_center[1])], 5, (255, 0, 0), -1)

                        plot_one_box(qrcode_xyxy, im0, label="qrcode", color=[255,0,0], line_thickness=1)
                    # print(temp_sop_list[1])
                    # for *xyxyo, conf, cls in reversed(det):
                    #     xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     cv2.circle(im0, [int(xywh[0]*img_x),int( xywh[1]*img_y)], 5, (0, 255, 0), -1) #green 辨識中心點在哪
                    # plot_one_box(tttt, im0, label=str(classs), color=[255,0,0], line_thickness=1)
                    # for i in range (len(tttt_xyxy0)):
                    #     plot_one_box(tttt_xyxy0[i], im0, label=tttt_text[i], color=[0,0,255], line_thickness=2)#colors[int(cls)]
                    try_icp_target = []
                    tp = []
                    temp_xmin=[]
                    temp_ymin=[]
                    temp_xmax=[]
                    temp_ymax=[]
                    for *xyxyo, conf, cls in reversed(det): #改成xyxyo 因為我要用xyxy
                        xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh 影像的正規化
                        xyxy = [(xywh[0]-xywh[2]/2)*img_x-objxmin ,(xywh[1]-xywh[3]/2)*img_y -objymin, (xywh[0]+xywh[2]/2)*img_x-objxmin , (xywh[1]+xywh[3]/2)*img_y-objymin]
                        tp.append([xywh[0],xywh[1],xywh[2],xywh[3]])
                        if  qrcode_xyxy[2] > int(xywh[0]*img_x) > qrcode_xyxy[0] and qrcode_xyxy[3]>int(xywh[1]*img_y) > qrcode_xyxy[1]: 
                            pass
                        else :
                            cv2.circle(im0, [int(xywh[0]*img_x),int( xywh[1]*img_y)], 5, (255, 0, 0), -1) #blue 辨識中心點在哪
                            xmin = xywh[0]-xywh[2]/2
                            ymin = xywh[1]-xywh[3]/2
                            xmax = xywh[0]+xywh[2]/2
                            ymax = xywh[1]+xywh[3]/2
                            temp_xmin.append(xmin)
                            temp_ymin.append(ymin)
                            temp_xmax.append(xmax)
                            temp_ymax.append(ymax)
                            xywh = xyxy2xywh_transfer(xyxy,[nw,nh],"xyxy2xywh")
                            xyxy = xyxy2xywh_transfer(xywh,[640,480],"xywh2xyxy")
                            xyxy_center = [xywh[0],xywh[1],0]
                            # xyxy_center = [(xyxy[0]+xyxy[2])/2,(xyxy[1]+xyxy[3])/2,0]

                            try_icp_target.append(xyxy_center)
                            # xyxy_center = [(xyxy[0]+xyxy[2])/2,(xyxy[1]+xyxy[3])/2,0] 
                    # print(len(try_icp_target))
                    # print("sop : ",len(try_icp_target),"hole : ",len(temp_xmin))
                    if len(temp_xmin) > 0 :
                        if (len(try_icp_source) - len(temp_xmin)) ==0: 
                            objxmin = min(temp_xmin)*img_x-20
                            objymin = min(temp_ymin)*img_y-20
                            objxmax = max(temp_xmax)*img_x+20
                            objymax = max(temp_ymax)*img_y+20
                            buf_tttt = [objxmin,objymin,objxmax,objymax] 
                            buf_icp_target = try_icp_target
                        elif (len(try_icp_source) - len(temp_xmin)) >= 2:#被擋住2洞
                            pass
                        # print(len(buf_icp_source))
                        ffff = buf_icp_target
                        tttt = buf_tttt
                        nw = (objxmax - objxmin)#物體的w
                        nh = (objymax - objymin)
                        # plot_one_box(tttt, im0, label=str(classs), color=[255,0,0], line_thickness=1)
                    cv2.putText(im0, str(classs)+" order : "+str(nl), (10, 40),cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                    ####
                    #### 藍框不會縮小 是這邊 objxmin 的問題 找一下  對比上面 9/23 
                    #### 要拒錄 鎖固 是9/23弄出來
                    ############classify type 

                    ############ icp 要確認     
                    ############ icp 要確認     
                    #source >> type
                    if len(temp_xmin) >1 :
                        target = np.asarray(try_icp_target) #try_icp_target
                        source = np.asarray(try_icp_source)  # yolo mode 1的 sop target 
                        pcd_target = o3d.geometry.PointCloud()
                        pcd_source = o3d.geometry.PointCloud()
                        source_temp = o3d.geometry.PointCloud()
                        pcd_target.points = o3d.utility.Vector3dVector(target)
                        pcd_source.points = o3d.utility.Vector3dVector(source)

                        threshold = 0.02
                        trans_init = np.asarray([[1,0,0,0], [0,1,0,0],  [0,0,1,0],   [0,0,0,1]])
                        reg_p2p = o3d.pipelines.registration.registration_icp(
                            pcd_source, pcd_target, threshold, trans_init,
                            o3d.pipelines.registration.TransformationEstimationPointToPoint(),o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 30))
                        source_temp = copy.deepcopy(pcd_source)
                        source_temp.transform(reg_p2p.transformation)
                        final_point = np.asarray(source_temp.points)

                        for i in range(len(temp_sop_list)):
                            source = [[temp_sop_list[i][2],temp_sop_list[i][3],0]]
                            pcd_source = o3d.geometry.PointCloud()
                            pcd_source.points = o3d.utility.Vector3dVector(source)
                            source_temp = copy.deepcopy(pcd_source)
                            source_temp.paint_uniform_color([0, 1, 0])
                            source_temp.transform(reg_p2p.transformation)
                            final_point = np.asarray(source_temp.points)

                            temp_sop_list[i][2] = final_point[0][0]
                            temp_sop_list[i][3] = final_point[0][1]
                    else:
                        print("如果被遮住剩一個洞 可能不準 就不匹配顯示上一次得洞")
                        # for i in range(int(len(tttt)/2)): # tttt icp
                        #     source = [[tttt[i*2],tttt[i*2+1],0]]
                        #     pcd_source = o3d.geometry.PointCloud()
                        #     pcd_source.points = o3d.utility.Vector3dVector(source)
                        #     source_temp = copy.deepcopy(pcd_source)
                        #     source_temp.paint_uniform_color([0, 1, 0])
                        #     source_temp.transform(reg_p2p.transformation)
                        #     final_point = np.asarray(source_temp.points)
                        #     tttt[i*2] =final_point[0][0]
                        #     tttt[i*2+1] =final_point[0][1]
                    ############ icp 要確認      
                    ############ 判斷出type 後給順序 
                    # print(nl,"nl")
                    plot_one_box(tttt, im0, label=str(classs), color=[255,0,0], line_thickness=1)
                    for *xyxyo, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxyo).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # xyxy = [(xywh[0]-xywh[2]/2)*img_x-objxmin ,(xywh[1]-xywh[3]/2)*img_y -objymin, (xywh[0]+xywh[2]/2)*img_x-objxmin , (xywh[1]+xywh[3]/2)*img_y-objymin]
                        # xywh = xyxy2xywh_transfer(xyxy,[nw,nh],"xyxy2xywh")
                        # xyxy = xyxy2xywh_transfer(xywh,[640,480],"xywh2xyxy")
                        # cv2.circle(im0, [int(xywh[0]*img_x),int( xywh[1]*img_y)], 5, (0, 255, 0), -1) #green 辨識中心點在哪
                        if  qrcode_xyxy[2] > int(xywh[0]*img_x) > qrcode_xyxy[0] and qrcode_xyxy[3]>int(xywh[1]*img_y) > qrcode_xyxy[1]: 
                            pass
                        else:
                            text = ""
                            find_range = 1.5
                            for i in range(len(temp_sop_list)):
                                xmin = (temp_sop_list[i][2]-temp_sop_list[i][4]/find_range)*(objxmax-objxmin) #(objxmax-objxmin)
                                ymin = (temp_sop_list[i][3]-temp_sop_list[i][5]/find_range)*(objymax-objymin) #(objymax-objymin)
                                xmax = (temp_sop_list[i][2]+temp_sop_list[i][4]/find_range)*(objxmax-objxmin) #(objxmax-objxmin)
                                ymax = (temp_sop_list[i][3]+temp_sop_list[i][5]/find_range)*(objymax-objymin) #(objymax-objymin)
                                t = [xmin+objxmin,ymin+objymin,xmax+objxmin,ymax+objymin]
                                r = [(temp_sop_list[i][2]*(objxmax-objxmin)+objxmin),(temp_sop_list[i][3]*(objymax-objymin)+objymin)]
                                # cv2.circle(im0, [int(r[0]),int(r[1])], 5, (0, 0, 255), -1) #red 轉換後點

                                # plot_one_box(t, im0, label=str(int(temp_sop_list[i][0])), color=[0,0,255], line_thickness=3)#colors[int(cls)]
                                
                                #反過來 QRCODE 中間有轉換後紅點 就OK
                                #轉換 無法90度 再看看


                                # hole 版本 會看到qrcode黑色 但不歪
                                # key = cv2.waitKey(1)
                                if temp_sop_list[i][6] == 0 and temp_sop_list[i][0] == nl: # nl
                                    rr = [(temp_sop_list[i][2]*(objxmax-objxmin)+objxmin),(temp_sop_list[i][3]*(objymax-objymin)+objymin)]
                                    cv2.circle(im0, [int(rr[0]),int(rr[1])], 5, (0, 0, 255), -1) #red 轉換後 nl 點

                                    if xmax+objxmin > xywh[0]*img_x > xmin+objxmin and ymax+objymin>xywh[1]*img_y > ymin+objymin: 
                                        cv2.putText(im0, str(int(temp_sop_list[i][0])), (int(xywh[0]*img_x-5), int(xywh[1]*img_y)-30), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)

                                        xyxy = [(xywh[0]-xywh[2]/2)*img_x ,(xywh[1]-xywh[3]/2)*img_y , (xywh[0]+xywh[2]/2)*img_x , (xywh[1]+xywh[3]/2)*img_y]
                                        tt = [(xywh[0]-xywh[2]/1.5)*img_x ,(xywh[1]-xywh[3]/1.5)*img_y , (xywh[0]+xywh[2]/1.5)*img_x , (xywh[1]+xywh[3]/1.5)*img_y]
                                        # plot_one_box(xyxy, im0, label=str(int(temp_sop_list[i][0])), color=[0,0,255], line_thickness=3)#colors[int(cls)]
                        #如果洞數量小於sop 2個 就用之前得物體大小    
                        if  qrcode_xyxy[2] > int(rr[0]) > qrcode_xyxy[0] and qrcode_xyxy[3]>int(rr[1]) > qrcode_xyxy[1]: 
                            key = cv2.waitKey(1)
                            # print(nl," 可以鎖 enter鎖固")
                            # cv2.rectangle(im0, (tt[0],tt[1]), (tt[2],tt[3]), (0,255,0), 6)

                            mask = cv2.rectangle(blk, (int(qrcode_xyxy[0]),int(qrcode_xyxy[1])), (int(qrcode_xyxy[2]),int(qrcode_xyxy[3])),color=(0,255,0), thickness=-1 ) 
                            im0 = cv2.addWeighted(im0, 1, mask, 0.5, 0)

                            # cv2.circle(im0, [int(qr_center[0]),int( qr_center[1])], 5, (0, 0, 255), -1)
                            # key = cv2.waitKey(1)
                            if key == 13 and yolo_mode == 2: # enter 27 esc
                                print('鎖固完成')
                                temp_sop_list[nl-1][6] = 1 
                                if nl < len(temp_sop_list):
                                    nl+=1
                                else :
                                    yolo_mode = 1
                                    print("都鎖完了!!!! 回辨識")
                                print(nl)
                        elif rr==[0,0]:
                            # print("有點在qrcode裡面")
                            pass
                        else :
                            # print("no")
                            mask = cv2.rectangle(blk, (int(qrcode_xyxy[0]),int(qrcode_xyxy[1])), (int(qrcode_xyxy[2]),int(qrcode_xyxy[3])),color=(0,0,255), thickness=-1 ) 
                            im0 = cv2.addWeighted(im0, 1, mask, 0.5, 0)
                            #

                            ##sop 版本 框框是歪的 
                            # if temp_sop_list[i][6] == 0 and temp_sop_list[i][0] == nl: # nl
                            #     # plot_one_box(xyxy, im0, label=label, color=[0,0,255], line_thickness=3)#colors[int(cls)]
                            #     tt = [xmin+objxmin,ymin+objymin,xmax+objxmin,ymax+objymin]
                            #     # qrcode 位置 改用 hole  不用sop
                            #     # 現在會話全 改一下
                            #     plot_one_box(tt, im0, label=str(int(temp_sop_list[i][0])), color=[0,255,0], line_thickness=3)#colors[int(cls)]
                                # 發現tt不太準 上面畫圈圈比較準 9/24
                            ##

                         
                if view_img:
                    # print("video")
                    cv2.imshow(str(p), im0)
                    key = cv2.waitKey(1)  # 1 millisecond
                    if key == 13 and yolo_mode == 1: # 13 enter 27 esc
                        yolo_mode = 2
                        print('into 鎖固')
                        nl = 1 #需要鎖固順序為1
                        # sop_dir = './Eclatorq/sop/labels/'+str(classs)+".txt"
                        # print(classs)
                        # f = open(sop_dir)
                        # for line in f.readlines():
                        #     #line = line.split(' ')
                        #     line = list(map(float, line.split(' ')))
                        #     try_icp_target.append([line[2],line[3],0])
                        #     temp_sop_list.append([line[0],line[1],line[2],line[3],line[4],line[5],0]) # 順序 , 種類 , x center ,y center , w , h 
                        # print("yolo mode 2 ")
                    elif yolo_mode == 2 and key ==27 :
                        yolo_mode = 1
                        nl = 0 # 初始化
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
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)

        if save_txt or save_img:
            # s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')
        return  save_dir
filename = os.path.basename(__file__)
filename=filename[17:len(filename)-3]
print("filename :",filename)

NUM_POINTS = 2048
NUM_CLASSES = 10
history_list=[]
model_list=[]
preds_list=[]
class_names_list= []
label_list=[]
color = [[0.5,0,0],'b','g','r','y','k','c','m']
def xyxy2xywh_transfer(temp_xy,imgyx,mode):
    if mode == 'xyxy2xywh':
        xywh = [((temp_xy[0]+temp_xy[2])/2)/imgyx[0],((temp_xy[1]+temp_xy[3])/2)/imgyx[1],abs((temp_xy[0]-temp_xy[2])) /imgyx[0],abs((temp_xy[1]-temp_xy[3])) /imgyx[1]]
        return xywh
    elif mode == 'xywh2xyxy':
        xyxy = [(temp_xy[0]-temp_xy[2]/2)*imgyx[0] ,(temp_xy[1]-temp_xy[3]/2)*imgyx[1] , (temp_xy[0]+temp_xy[2]/2)*imgyx[0] , (temp_xy[1]+temp_xy[3]/2)*imgyx[1]]
        return xyxy





def classify2tensor ():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best_298.pt', help='model.pt path(s)') # yolov7.pt
    parser.add_argument('--source', type=str, default='2', help='source')  # file/folder  inference/images  , 0 for webcam
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
    parser.add_argument('--classification_type', action='store_true',default='PN', help='don`t trace model')
    opt = parser.parse_args()
    # print(opt.source)
    cs.detect(opt) 

#fit_list = [x_train,x_train]


if __name__ == "__main__":
    cs = classify()
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
