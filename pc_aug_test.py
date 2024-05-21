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
DATA_DIR = "./Eclatorq/type"
npy_path = "./Eclatorq/npy/"
save_name_p = "all_obj.npy"
save_name_l = "all_label.npy"

import matplotlib
import open3d as o3d
matplotlib.use('TkAgg')
def old_parse_dataset(num_points,DATA_DIR):
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_names = []
    train_name = "yolo"
    print("train_name :",train_name)
    #return train_name

    folders = glob.glob(os.path.join(DATA_DIR, "*"))# Type_C
    print(folders)
    for i, folder in enumerate(folders):
        temp = os.path.basename(folder)
        print("processing class: {}".format(temp))
        class_names.append(temp)
        train_files = glob.glob(os.path.join(folder, "*"))
        if train_name == "yolo":
            for f in train_files:
                # print(f)
                obj_pc = []
                obj_hole = []
                f = open(f)
                for line in f.readlines():
                    line = list(map(float, line.split(' ')))
                    xyxy = xyxy2xywh_transfer([float(line[1]),float(line[2]),float(line[3]),float(line[4])],[640,480],'xywh2xyxy')
                    # print(xyxy)
                    obj_pc = ([round((xyxy[0]+xyxy[2])/2, 3),round((xyxy[1]+xyxy[3])/2, 3),0])
                    obj_hole.append(obj_pc)
                    # print(obj_pc)
                    # cv2.rectangle(img, [int(self.xyxy[0]),int(self.xyxy[1])], [int(self.xyxy[2]),int(self.xyxy[3])], (0, 0, 255), 1)
                f.close
                if len(obj_hole)<20 :
                    add_num = 20-len(obj_hole)
                    # print(add_num)
                    for j in range(add_num):
                        obj_hole.append([0,0,0]) # 代表填充的
                # print(obj_hole)
                train_points.append(obj_hole)
                train_labels.append(i)

    # np.save("./Eclatorq/npy/"+save_name_p, train_points)
    # np.save("./Eclatorq/npy/"+save_name_l, train_labels)
    class_names = np.array(class_names)
    return class_names,train_points,train_labels    
def old_yoma_pc_aug (train_points):#(1,20,3)
    aug_pc=  []
    # theta = 0
    # rot_matrix = np.array([
    #     [np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta)), 0],
    #     [np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta)), 0],
    #     [0, 0, 1]])

    random_num  = np.sort(np.random.choice(len(train_points), size=36, replace=False))
    print(random_num)
    # for j in range(len(random_num)) : # 36
    #     # print(j)
    #     obj_pc = train_points[random_num[j]]
    #     test = np.transpose(obj_pc) # (20,3) t0 (3,20)

    for i in range(10): #360度
        for j in range(len(random_num)) :
            # print(i*36+(j+1))# 0 ~ 10 
            # print(j)
            obj_pc = train_points[random_num[j]]
            test = np.transpose(obj_pc) # (20,3) t0 (3,20)
            theta = (i*36+(j+1))
            # print(j*i)
            rot_matrix = np.array([
            [np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta)), 0],
            [np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta)), 0],
            [0, 0, 1]])
            new_point = np.dot(rot_matrix, test)
            plt.scatter(new_point[0], new_point[1],color='red')

            aug_point=np.transpose(new_point)
            aug_pc.append(aug_point)
    return aug_pc
def yoma_pc_aug (train_points):#(1,20,3)
    aug_pc=  []
    get_size = 10
    random_num  = np.sort(np.random.choice(len(train_points), size=get_size, replace=False))
    print(random_num)

    for i in range(get_size): #360度 每1個data 轉10度 >> 360 data
        e_pc = train_points[random_num[i]]
        e_p = np.transpose(e_pc)  # (20,3) t0 (3,20)
        # plt.scatter(e_p[0], e_p[1], color='b') # 原圖 找重心用
        aug_point = []
        for k in range(20):
            if e_pc[k] != [0, 0, 0]:
                aug_point.append(e_pc[k])
        source = np.asarray(aug_point)
        pp = o3d.geometry.PointCloud()
        pp.points = o3d.utility.Vector3dVector(source)
        org_center = pp.get_center()
        # print(org_center)
        plt.scatter(org_center[0], org_center[1], color='g') # 原圖g的重心

        # for j in range(30) :
        for j in list(range(2,180,2)):#[30,60]
            obj_pc = train_points[random_num[i]]
            aa=[]
            for k in range(20):
                if obj_pc[k] != [0, 0, 0]:
                    aa.append(obj_pc[k])
            test = np.transpose(aa) # (20,3) t0 (3,20)
            theta = j
            rot_matrix = np.array([
            [np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta)), 0],
            [np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta)), 0],
            [0, 0, 1]])
            new_point = np.dot(rot_matrix, test)
            # plt.scatter(new_point[0], new_point[1],color='red') # 沒-重心前

            aug_point=np.transpose(new_point)

            source = np.asarray(aug_point)
            p = o3d.geometry.PointCloud()
            p.points = o3d.utility.Vector3dVector(source)

            cc = p.get_center()
            # print(cc, 'cc')
            error = cc-org_center
            # print(error)
            # plt.scatter(cc[0], cc[1], color='g') # 每個aug的重心
            # print("before",np.shape(aug_point))
            target_num = 20

            for a in range(len(aug_point)):
                if aug_point[a].any():
                    aug_point[a][0]=aug_point[a][0]-error[0]
                    aug_point[a][1]=aug_point[a][1]-error[1]
            # print(np.shape(aug_point))

            for b in range(target_num-len(aug_point)):
                aug_point = list(aug_point)
                aug_point.append([0,0,0])
            # print(np.shape(aug_point))
            plt.scatter(new_point[0], new_point[1],color='red') # aug-重心後

            aug_pc.append(aug_point)

            # print(aug_pc)

    return np.array(aug_pc)
def augparse_dataset(num_points,DATA_DIR):
    all_train_points = []
    all_train_labels = []

    class_names = []
    train_name = "yolo"
    print("train_name :",train_name)
    #return train_name

    folders = glob.glob(os.path.join(DATA_DIR, "*"))# Type_C
    print(folders)
    for i, folder in enumerate(folders):
        train_points = []
        train_labels = []
        temp = os.path.basename(folder)
        print("processing class: {}".format(temp))
        class_names.append(temp)
        train_files = glob.glob(os.path.join(folder, "*"))
        if train_name == "yolo":
            for f in train_files:
                # print(f)
                obj_pc = []
                obj_hole = []
                f = open(f)
                for line in f.readlines():
                    line = list(map(float, line.split(' ')))
                    xyxy = xyxy2xywh_transfer([float(line[1]),float(line[2]),float(line[3]),float(line[4])],[640,480],'xywh2xyxy')
                    # print(xyxy)
                    obj_pc = ([round((xyxy[0]+xyxy[2])/2, 3),round((xyxy[1]+xyxy[3])/2, 3),0])
                    obj_hole.append(obj_pc)
                    # print(obj_pc)
                    # cv2.rectangle(img, [int(self.xyxy[0]),int(self.xyxy[1])], [int(self.xyxy[2]),int(self.xyxy[3])], (0, 0, 255), 1)
                f.close
                if len(obj_hole)<20 :
                    add_num = 20-len(obj_hole)
                    # print(add_num)
                    for j in range(add_num):
                        obj_hole.append([0,0,0]) # 代表填充的
                # print(obj_hole)
                train_points.append(obj_hole)
                train_labels.append(i)
    ###
        aug_pc = yoma_pc_aug(train_points) # yoma_pc_aug
        # print(np.shape(train_labels))
        all_train_points.append(aug_pc)
        tttt = train_labels[0]
        print(tttt)
        all_train_labels.append([tttt]*len(aug_pc))

    temp_shape =np.shape(all_train_points)
    print(temp_shape)
    # print(np.shape(all_train_labels))

    all_train_points = np.array(all_train_points).reshape((temp_shape[0]*temp_shape[1],temp_shape[2],temp_shape[3]))
    all_train_labels = np.array(all_train_labels).reshape((temp_shape[0]*temp_shape[1]))
    print(np.shape(all_train_points))

    print(np.shape(all_train_labels))
    ###
    np.save("./Eclatorq/npy/"+save_name_p, all_train_points)
    np.save("./Eclatorq/npy/"+save_name_l, all_train_labels)
    class_names = np.array(class_names)
    return class_names,train_points,train_labels   
def load_dataset(DATA_DIR):
    print("load")
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_names = []
    #return train_name


    if not os.path.exists(npy_path+save_name_p):
        print("not exists")
        class_names,train_points,train_labels = augparse_dataset(2048,DATA_DIR)

    else:
        print("exists")
        folders = glob.glob(os.path.join(DATA_DIR, "*"))
        for i, folder in enumerate(folders):
            temp = os.path.basename(folder)
            print("processing class: {}".format(temp))
            class_names.append(temp)
            train_points = np.load(npy_path+save_name_p, allow_pickle=True)
            train_labels = np.load(npy_path+save_name_l, allow_pickle=True)
            # print(train_points)
        class_names = np.array(class_names)
    return class_names,train_points,train_labels    

NUM_POINTS = 2048
NUM_CLASSES = 10
history_list=[]
model_list=[]
preds_list=[]
class_names_list= []
color = [[0.5,0,0],'b','g','r','y','k','c','m']

def xyxy2xywh_transfer(temp_xy,imgyx,mode):
    if mode == 'xyxy2xywh':
        xywh = [((temp_xy[0]+temp_xy[2])/2)/imgyx[0],((temp_xy[1]+temp_xy[3])/2)/imgyx[1],abs((temp_xy[0]-temp_xy[2])) /imgyx[0],abs((temp_xy[1]-temp_xy[3])) /imgyx[1]]
        return xywh
    elif mode == 'xywh2xyxy':
        xyxy = [(temp_xy[0]-temp_xy[2]/2)*imgyx[0] ,(temp_xy[1]-temp_xy[3]/2)*imgyx[1] , (temp_xy[0]+temp_xy[2]/2)*imgyx[0] , (temp_xy[1]+temp_xy[3]/2)*imgyx[1]]
        return xyxy



#fit_list = [x_train,x_train]
aug_pc=  []
def pc_aug (obj_pc):#(1,20,3)
    aug_pc=  []
    # theta = 0
    # rot_matrix = np.array([
    #     [np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta)), 0],
    #     [np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta)), 0],
    #     [0, 0, 1]])
    test = np.transpose(obj_pc) # (20,3) t0 (3,20)
    for i in range(360): #360度
        theta = i
        rot_matrix = np.array([
        [np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta)), 0],
        [np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta)), 0],
        [0, 0, 1]])
        new_point = np.dot(rot_matrix, test)
        plt.scatter(new_point[0], new_point[1],color='red')

        aug_point=np.transpose(new_point)
        aug_pc.append(aug_point)
    return aug_pc
# 定义点坐标

# 进行矩阵乘法运算
# new_point = np.dot(rot_matrix, point)

if __name__ == "__main__":
    class_names,train_points,train_labels =load_dataset(DATA_DIR)


    # plt.show()

