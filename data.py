####################################
######## 20230608 整合 可以執行 但需要做初始化 資料夾設定 
######## 通用化path 都抓相對路徑 建立path_list 如果沒有要mkdir
######## augmentations_yoma 設定一張圖片 變成幾張 之後要調整 照片好看一點
####################################
####################################
######## 20231105 calss化  可以用不同資料 
######## init 應該要多加 albumentations 的參數 
######## 目前可以設定名稱 數量 
######## 各副程式優化 新增error function 
######## 過濾轉換出錯的檔案  自動清除之前資料(只保留input data)
######## 其他py使用 >> from data import augmentations_yoma as aug_yoma 
######## aug = aug_yoma(name = 'hole',augment_num = 10 ,classes = classes,category_id_to_name = category_id_to_name)
####################################
import shutil
import random
import os
import glob
import pickle
import xml.etree.ElementTree as ET
from os import listdir, getcwd
from os.path import join
import cv2
from matplotlib import pyplot as plt
import numpy as np
import albumentations as A
from PIL import Image
import glob
import os
import pickle
from os.path import join
from pascal_voc_writer import Writer

def del_file(path):
    for i in os.listdir(path):
        file_data = path + "\\" + i
        os.remove(file_data)
def getImagesInDir(dir_path):
    image_list = []
    for filename in glob.glob(dir_path + '/*.jpg'):
        image_list.append(filename)

    return image_list
def error_fuc(error_list):
    output_error = ''
    error_flag  = np.unique(error_list)
    if error_flag.size <= 0:
        error_flag =[0]
    for i in range(len(error_flag)):
        # print(error_message[str(error_flag[i])])
        error_detail = eval('error_'+str(error_flag[i]))
        # print(error_detail)
        output_error += ' '+error_message[str(error_flag[i])]+"\n"
        for j in error_detail : # list有東西會 print 出  沒東西就不會執行
            output_error += ' '+j 
    return print(output_error)
def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
import yaml

class augmentations_yoma ():
    def __init__(self,name = 'hole',augment_num=10,classes=[],category_id_to_name={}):
        self.classes = classes
        self.category_id_to_name = category_id_to_name
        self.augment_num = augment_num
        self.data_name = name
        self.cwd = getcwd()
        # class seting 

        #輸入
        self.input_image_path = self.cwd + '/yoma_data/'+self.data_name+'/input_images'
        self.input_voc_path = self.cwd + '/yoma_data/'+self.data_name+'/input_voc'
        #整理
        self.org_voc_dir_path = self.cwd + '/yoma_data/'+self.data_name+'/org_voc/'
        self.org_img_dir_path = self.cwd + '/yoma_data/'+self.data_name+'/org_images/'
        #增強
        self.aug_voc_dir_path = self.cwd + '/yoma_data/'+self.data_name+'/voc/'
        self.aug_img_dir_path = self.cwd + '/yoma_data/'+self.data_name+'/images/'
        #轉換
        self.output_label_path = self.cwd + '/yoma_data/'+self.data_name+'/label/'
        #產生yolo 所需
        self.train_image_path = os.path.join(self.cwd, "datasets/defect/images/train/")
        self.train_label_path = os.path.join(self.cwd, "datasets/defect/labels/train/")
        self.val_image_path = os.path.join(self.cwd, "datasets/defect/images/val/")
        self.val_label_path = os.path.join(self.cwd, "datasets/defect/labels/val/")
        self.test_image_path = os.path.join(self.cwd, "datasets/defect/images/test/")
        self.test_label_path = os.path.join(self.cwd, "datasets/defect/labels/test/")

        self.list_train = os.path.join(self.cwd, "datasets/defect/train.txt")
        self.list_val = os.path.join(self.cwd, "datasets/defect/val.txt")
        self.train_cache = os.path.join(self.cwd, "datasets/defect/train.cache")
        self.val_cache = os.path.join(self.cwd, "datasets/defect/val.cache")
        self.list_test = os.path.join(self.cwd, "datasets/defect/test.txt")

        self.train_percent = 0.6
        self.val_percent = 0.2
        self.test_percent = 0.2
        self.clearfile()
        self.mkdir()
    def run(self):
        self.set_ymal()
        self.count_num2jpg(self.input_image_path)
        image_paths = getImagesInDir(self.org_img_dir_path)
        # print("input_img_list ",image_paths)
        print('augment_num : ',self.augment_num)
        print("____________________Processing____________________")
        for image_path in image_paths:
            self.bboxes = []
            self.category_ids = []
            classes_list = []

            #print(image_path)
            temp = os.path.splitext(image_path)[0]
            name = os.path.split(temp)[1]
            print("img_name : ",name)

            bboxes = self.read_voc(self.org_voc_dir_path, image_path)

            self.aug_yoma(image_path,bboxes,name)
        # print("Finished augmentations_yoma processing " )
        self.voc2yolotxt()
        # print("Finished transfrom processing " )
        result = self.main()
        # print("Finished to_yolo_train processing " )
        print("_______________________Done_______________________")
        print("______________________ERROR_______________________\n")
        error_fuc(error_list)
        print("______________________ERROR_______________________")
        print(result)
    def set_ymal(self):

        d = {'train':'./datasets/defect/train.txt','val':'./datasets/defect/val.txt','test':'./datasets/defect/test.txt',
        'nc':len(self.classes),'names':self.classes}
        print('set yaml' ,'./data/yoma.yaml',self.classes )
        with open('./data/yoma.yaml', 'w') as f:
            yaml.dump(d, f)

    def clearfile(self):
        if os.path.exists(self.list_train):
            os.remove(self.list_train)
        if os.path.exists(self.list_val):
            os.remove(self.list_val)
        if os.path.exists(self.list_test):
            os.remove(self.list_test)# train.cache
        if os.path.exists(self.train_cache):
            os.remove(self.train_cache)
        if os.path.exists(self.val_cache):
            os.remove(self.val_cache)

    def mkdir(self):
        need1  = self.input_image_path 
        need2  = self.input_voc_path 
        need3  = self.org_voc_dir_path 
        need4  = self.org_img_dir_path 
        need5  = self.aug_voc_dir_path 
        need6  = self.aug_img_dir_path 
        need7  = self.output_label_path
        need8  = self.train_image_path 
        need9  = self.train_label_path 
        need10 = self.val_image_path 
        need11 = self.val_label_path 
        need12 = self.test_image_path 
        need13 = self.test_label_path 
        last_need = "need13" # 設定最後路徑
        #新增參數下面要更改
        need_path = []

        times = eval(last_need[4:(len(last_need)+1)])
        for i in range(times):
            need_path_buf = eval("need"+str(i+1))
            if need_path_buf != '':
                need_path.append(need_path_buf)
            else:
                pass
        necessary_env = os.path.split(os.getcwd())[1]
        print("env : ",necessary_env)
        for i in range(len(need_path)):
            if not os.path.exists(need_path[i]):
                print("mkdir : "+ str(need_path[i]))
                os.makedirs(need_path[i])
            else :
                print("exists : "+ str(need_path[i]))
                if os.path.split(need_path[i])[1] == 'input_images' or os.path.split(need_path[i])[1] == 'input_voc' :
                    # print("不能刪除input")
                    pass
                else:
                    del_file(need_path[i])
        # if necessary_env =='yolov7':
        #     print("This Project Must Used By yolov7 ")

        # else:
        #     print("This Project Must Used By yolov7 ")
    def count_num2jpg(self,dir_path):
        files = []
        for ext in ('*.png', '*.jpg'):
            # print(ext)
            files.extend(glob.glob(os.path.join(dir_path, ext)))
            print(ext[2:5],"num :",len(files))
        img = []
        if len(files) > 0 :
            for i in range(len(files)): # 亂七八糟檔名 轉換成數字 跟固定 jpg
                img_name = os.path.splitext(os.path.basename(files[i]))[0]
                # print(img_name)
                im = Image.open(files[i])
                im_RGBA = im.convert("RGB")
                if not os.path.exists(self.input_voc_path + '/' +img_name + '.xml'):
                    print("label error")
                    error_list.append(4)
                else :
                    im_RGBA.save(self.org_img_dir_path+str(i)+'.jpg')
                    in_file = open(self.input_voc_path + '/' + img_name + '.xml')
                    tree = ET.parse(in_file)
                    tree.write(self.org_voc_dir_path+str(i)+'.xml')
                # print(in_file) #<_io.TextIOWrapper name='D:\\pycharm\\yolov7/yoma_data/input_voc//8.xml' mode='r' encoding='cp950'>

                # tree = ET.parse(in_file)
                # tree.write(self.org_voc_dir_path+str(i)+'.xml')
        else :
            error_list.append(3)
    def read_voc(self,dir_path, image_path):
        basename = os.path.basename(image_path)
        basename_no_ext = os.path.splitext(basename)[0]


        in_file = open(dir_path + '/' + basename_no_ext + '.xml')
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in self.classes or int(difficult)==1:
                continue
            cls_id = self.classes.index(cls)
            self.category_ids.append(cls_id)
            # classes_list.append(classes[cls_id])
            #print(cls_id,classes[cls_id])
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
            self.bboxes.append(b)
        return self.bboxes
    def write_voc(self,img,bboxes,category_ids,category_id_to_name,name):

        # create pascal voc writer (image_path, width, height)
        #print(bboxes)
        #print(category_ids)
        for num in range(len(img)):
            save_name = name +'_'+str(num)
            height,width, _ = img[num].shape
            writer = Writer(self.aug_voc_dir_path+save_name+'.jpg', width,height)
            #print(num,"num")
            temp  = category_ids[num]
            #print(temp)
            box = bboxes[num]
            for i in range(len(temp)):
                save_xmin = box[i][0]
                save_ymin = box[i][1]
                save_xmax = box[i][2]
                save_ymax = box[i][3]
                #print(category_id_to_name[temp[i]])
                writer.addObject(category_id_to_name[temp[i]], save_xmin, save_ymin, save_xmax, save_ymax)

            # write to file
            writer.save(self.aug_voc_dir_path+save_name+'.xml')
            cv2.imwrite(self.aug_img_dir_path+save_name+'.jpg', img[num])
    def aug_yoma(self,imgaes_list,bboxes_list,name):
        # print("start augmentations ")
        cv2_image = cv2.imread(imgaes_list)
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

        bboxes = bboxes_list
        buf_boxnum = len(bboxes)
        # print("buf_boxnum : ",len(bboxes))
        # Pascal_voc (x_min, y_min, x_max, y_max), YOLO, COCO

        transform = A.Compose(
            [
                A.Resize(width=1920, height=1080),
                # A.RandomCrop(width=1280, height=720),
                A.Rotate(limit=15, p=0.9, border_mode=cv2.BORDER_CONSTANT),
                A.HorizontalFlip(p=0.4),
                A.VerticalFlip(p=0.3),
                A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
                A.OneOf([A.Blur(blur_limit=3, p=0.5),A.ColorJitter(p=0.5),], p=1.0),
                # A.Resize(width=1280, height=720),
            ], bbox_params=A.BboxParams(format="pascal_voc", min_area=2048,
                                        min_visibility=0.3, label_fields=['category_ids'])
        )

        images_list = [cv2_image]
        saved_bboxes = [bboxes]
        augmented_label_ids_list = [self.category_ids]
        for i in range(self.augment_num):
            augmentations = transform(image=cv2_image, bboxes=bboxes,category_ids=self.category_ids)
            # augmented_img = augmentations["image"]
            # augmented_label_ids = augmentations["category_ids"]
            # print(augmented_label_ids)
            # visualize(augmentations['image'],augmentations['bboxes'],augmentations['category_ids'],category_id_to_name,)
            # plt.show()
            # if len(augmentations["bboxes"]) == 0:
            #     continue

            images_list.append(augmentations["image"])
            augmented_label_ids_list.append(augmentations["category_ids"])
            saved_bboxes.append(augmentations["bboxes"])
        if len(saved_bboxes[1]) != buf_boxnum : # len(saved_bboxes[1]) = 產生出的第一個 來判斷跟原本label 數量有差嗎
            # print(len(saved_bboxes[1]), buf_boxnum )
            error_list.append(2)
            print(name," : box_num error ","augmentations :",len(saved_bboxes[1]),"org :", buf_boxnum )
            error_2.append(name)
            self.write_voc([images_list[0]],[saved_bboxes[0]],[augmented_label_ids_list[0]],self.category_id_to_name,name)
            # 加上 原始版本

        else:
            # print(images_list)
            # print(augmented_label_ids_list)
            # print(saved_bboxes)

            self.write_voc(images_list,saved_bboxes,augmented_label_ids_list,self.category_id_to_name,name)
            # plot_examples(images_list, saved_bboxes,augmented_label_ids_list,category_id_to_name)
    def voc2yolotxt(self):

        # print(output_path)
        if not os.path.exists(self.output_label_path):
            os.makedirs(self.output_label_path)

        image_paths = getImagesInDir(self.aug_img_dir_path)
        # print(image_paths)
        list_file = open(self.aug_voc_dir_path + '.txt', 'w')
        #print(list_file)

        for image_path in image_paths:
            list_file.write(self.aug_img_dir_path + '\n')
            self.convert_annotation(self.aug_voc_dir_path, self.output_label_path,image_path)
        list_file.close()
    def convert_annotation(self,dir_path, output_path, image_path):
        basename = os.path.basename(image_path)
        basename_no_ext = os.path.splitext(basename)[0]


        in_file = open(dir_path + '/' + basename_no_ext + '.xml')
        out_file = open(output_path + basename_no_ext + '.txt', 'w')
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        tt = root.find('object')
        # print(tt)
        if tt==None:
            error_list.append(1)
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in self.classes or int(difficult)==1:
                print("difficult == 1")
                continue

            cls_id = self.classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w,h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    def main(self):# 產生 yolo 所需

        file_train = open(self.list_train, 'w')
        file_val = open(self.list_val, 'w')
        file_test = open(self.list_test, 'w')

        total_txt = os.listdir(self.output_label_path)
        num_txt = len(total_txt)
        list_all_txt = range(num_txt)

        num_train = int(num_txt * self.train_percent)
        num_val = int(num_txt * self.val_percent)
        num_test = num_txt - num_train - num_val

        train = random.sample(list_all_txt, num_train)
        # train从list_all_txt取出num_train个元素
        # 所以list_all_txt列表只剩下了这些元素
        val_test = [i for i in list_all_txt if not i in train]
        # 再从val_test取出num_val个元素，val_test剩下的元素就是test
        val = random.sample(val_test, num_val)
        bmp = '.bmp'
        jpg = '.jpg'
        for i in list_all_txt:
            name = total_txt[i][:-4]

            srcImage = self.aug_img_dir_path + name + jpg
            srcLabel = self.output_label_path + name + ".txt"

            if i in train:
                # print(name)
                dst_train_Image = self.train_image_path + name + jpg
                dst_train_Label = self.train_label_path + name + '.txt'
                shutil.copyfile(srcImage, dst_train_Image)
                shutil.copyfile(srcLabel, dst_train_Label)
                file_train.write(dst_train_Image + '\n')
            elif i in val:
                dst_val_Image = self.val_image_path + name + jpg
                dst_val_Label = self.val_label_path + name + '.txt'
                shutil.copyfile(srcImage, dst_val_Image)
                shutil.copyfile(srcLabel, dst_val_Label)
                file_val.write(dst_val_Image + '\n')
            else:
                dst_test_Image = self.test_image_path + name + jpg
                dst_test_Label = self.test_label_path + name + '.txt'
                shutil.copyfile(srcImage, dst_test_Image)
                shutil.copyfile(srcLabel, dst_test_Label)
                file_test.write(dst_test_Image + '\n')

        file_train.close()
        file_val.close()
        file_test.close()#        

        result = "Final_num : {},train_num ： {}, val_num ： {}, test_num ： {}".format(num_txt,len(train), len(val), len(val_test) - len(val))
        return     result


# classes = ['hole','obj','wrench']
# category_id_to_name = {0: classes[0], 1: classes[1],2: classes[2]}
# error_list = []
# error_message= {'0':'All Fine','1':'voc2yolo voc dont have label(augmentations error)','2':'box_num error','3':'inputfile is empty ','4':'img no have label '}
# error_0,error_1,error_2,error_3,error_4 =[list()for i in range (len(error_message))]
# ttt = augmentations_yoma(name = 'test',augment_num = 2 ,classes = classes,category_id_to_name = category_id_to_name)
# ttt.run()