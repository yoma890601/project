####################################
######## 20230608整合 可以執行 但需要做初始化 資料夾設定 
######## 通用化path 都抓相對路徑 建立path_list 如果沒有要mkdir
######## augmentations_yoma 設定一張圖片 變成幾張 之後要調整 照片好看一點
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
from os import listdir, getcwd
import glob
import os
import pickle
import xml.etree.ElementTree as ET
from os.path import join
from pascal_voc_writer import Writer
# D:\pycharm\yolov7/yoma_data/org_images/1.png
error_list = []


error_message= {'0':'All Fine','1':'voc2yolo voc dont have label(augmentations error)','2':'box_num error'}
error_0,error_1,error_2 =[list()for i in range (len(error_message))]

classes = ['hole','gg']
category_id_to_name = {0: classes[0], 1: classes[1]}
augment_num = 10

def count_num2png(dir_path):
    files = []
    for ext in ('*.png', '*.jpg'):
        # print(ext)
        files.extend(glob.glob(os.path.join(dir_path, ext)))

            # im = Image.open('image.jpg')
            # im.save('image.png')

    print(ext[2:5],"num :",len(files))
    img = []
    for i in range(len(files)):
        img_name = os.path.splitext(os.path.basename(files[i]))[0]
        # print(img_name)
        im = Image.open(files[i])
        im_RGBA = im.convert("RGB")

        im_RGBA.save(org_img_dir_path+str(i)+'.jpg')

        in_file = open(input_voc_path + '/' + img_name + '.xml')
        # print(in_file) #<_io.TextIOWrapper name='D:\\pycharm\\yolov7/yoma_data/input_voc//8.xml' mode='r' encoding='cp950'>

        tree = ET.parse(in_file)
        tree.write(org_voc_dir_path+str(i)+'.xml')
        # img.append(im)
        # im.close()
        # os.remove(files[i])
    # for i in range  (len(img)):
    #     in_file = open(input_voc_path + '/' + os.path.splitext(os.path.basename(files[i]))[0] + '.xml')
    #     print(in_file)
    #     tree = ET.parse(in_file)
    #     tree.write(org_voc_dir_path+str(i)+'.xml')
        # imgg = img[i]
        # imgg.save(org_img_dir_path+str(i)+'.png')

def plot_examples(images, bboxes=None,augmented_label_ids_list=None,category_id_to_name=None):
    fig = plt.figure(figsize=(15, 15))
    columns = 4
    rows = 5

    for i in range(1, len(images)):

        if bboxes is not None:
            img = visualize_bbox_y(images[i - 1], bboxes[i - 1], augmented_label_ids_list[i-1],category_id_to_name)
        else:
            img = images[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    # plt.show()
# From https://albumentations.ai/docs/examples/example_bboxes/
def visualize_bbox_y(img, bbox, augmented_label_ids_list,category_id_to_name, color=(255, 0, 0)):
    """Visualizes a single bounding box on the image"""

    for box_num in range(len(bbox)):
        x_min, y_min, x_max, y_max = map(int, bbox[box_num])
        idd = augmented_label_ids_list[box_num]
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 5)
        class_name = category_id_to_name[idd]
        #print(class_name,"class_name")
        retval, baseLine = cv2.getTextSize(class_name,fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=2, thickness=5)

        #((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)

        cv2.putText(img,text=class_name,org=(x_min, y_min - baseLine),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
        fontScale=2, color=(0, 0, 255), lineType=cv2.LINE_AA,)
    return img
def getImagesInDir(dir_path):
    image_list = []
    for filename in glob.glob(dir_path + '/*.jpg'):
        image_list.append(filename)

    return image_list
def read_voc(dir_path, image_path):
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
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        category_ids.append(cls_id)
        # classes_list.append(classes[cls_id])
        #print(cls_id,classes[cls_id])
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
        bboxes.append(b)
    #print(bboxes)
    #print(len(bboxes))
def write_voc(img,bboxes,category_ids,category_id_to_name,name):

    # create pascal voc writer (image_path, width, height)
    #print(bboxes)
    #print(category_ids)
    for num in range(len(img)):
        save_name = name +'_'+str(num)
        height,width, _ = img[num].shape
        writer = Writer(cwd + '/yoma_data/voc/'+save_name+'.jpg', width,height)
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
        writer.save(cwd + '/yoma_data/voc/'+save_name+'.xml')
        cv2.imwrite(cwd + '/yoma_data/images/'+save_name+'.jpg', img[num])
def aug_yoma(imgaes_list,bboxes_list,name):
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
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.1),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
            A.OneOf([A.Blur(blur_limit=3, p=0.5),A.ColorJitter(p=0.5),], p=1.0),
            # A.Resize(width=1280, height=720),
        ], bbox_params=A.BboxParams(format="pascal_voc", min_area=2048,
                                    min_visibility=0.3, label_fields=['category_ids'])
    )

    images_list = [cv2_image]
    saved_bboxes = [bboxes]
    augmented_label_ids_list = [category_ids]
    for i in range(augment_num):
        augmentations = transform(image=cv2_image, bboxes=bboxes,category_ids=category_ids)
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
    else:
        write_voc(images_list,saved_bboxes,augmented_label_ids_list,category_id_to_name,name)
        plot_examples(images_list, saved_bboxes,augmented_label_ids_list,category_id_to_name)

def augmentations_yoma(): #廢棄 drop
    cwd = getcwd()
    org_voc_dir_path = cwd + '/' +'yoma_data'+ '/org_voc/'
    org_img_dir_path = cwd + '/' +'yoma_data' +'/org_images/'
    aug_voc_dir_path = cwd + '/' +'yoma_data'+ '/voc/'
    aug_img_dir_path = cwd + '/' +'yoma_data' +'/images/'
    image_paths = getImagesInDir(org_img_dir_path)
    print("input_img_list",image_paths)
    for image_path in image_paths:
        bboxes = []
        category_ids = []
        classes_list = []
        temp = os.path.splitext(image_path)[0]
        name = os.path.split(temp)[1]
        print("img_name : ",name)
        read_voc(org_voc_dir_path, image_path)
        aug_yoma(image_path,bboxes,name)
    print(" Finished processing " )
# ___________________________________________________________________

def _init_yoma (username = 'yoma'):#目前沒用 因為醫用就抓不到檔案

    # 原始路径
    image_original_path = "./"+username+"_data/images/"
    label_original_path = "./"+username+"_data/label/"

    cur_path = os.getcwd()
    print(cur_path)

    train_image_path = os.path.join(cur_path, "datasets/defect/images/train/")
    train_label_path = os.path.join(cur_path, "datasets/defect/labels/train/")
    print(train_image_path)
    # 验证集路径
    val_image_path = os.path.join(cur_path, "datasets/defect/images/val/")
    val_label_path = os.path.join(cur_path, "datasets/defect/labels/val/")

    # 测试集路径
    test_image_path = os.path.join(cur_path, "datasets/defect/images/test/")
    test_label_path = os.path.join(cur_path, "datasets/defect/labels/test/")

    # 训练集目录
    list_train = os.path.join(cur_path, "datasets/defect/train.txt")
    list_val = os.path.join(cur_path, "datasets/defect/val.txt")
    train_cache = os.path.join(cur_path, "datasets/defect/train.cache")
    val_cache = os.path.join(cur_path, "datasets/defect/val.cache")
    list_test = os.path.join(cur_path, "datasets/defect/test.txt")
    train_percent = 0.6
    val_percent = 0.2
    test_percent = 0.2
     ###   voc 2 yolo
    classes = ['hole']
def del_file(path):
    for i in os.listdir(path):
        file_data = path + "\\" + i
        os.remove(file_data)
def mkdir():
    if not os.path.exists(train_image_path):
        os.makedirs(train_image_path)
    else:
        del_file(train_image_path)
    if not os.path.exists(train_label_path):
        os.makedirs(train_label_path)
    else:
        del_file(train_label_path)

    if not os.path.exists(val_image_path):
        os.makedirs(val_image_path)
    else:
        del_file(val_image_path)
    if not os.path.exists(val_label_path):
        os.makedirs(val_label_path)
    else:
        del_file(val_label_path)

    if not os.path.exists(test_image_path):
        os.makedirs(test_image_path)
    else:
        del_file(test_image_path)
    if not os.path.exists(test_label_path):
        os.makedirs(test_label_path)
    else:
        del_file(test_label_path)

    if not os.path.exists(org_img_dir_path):
        os.makedirs(org_img_dir_path)
    else:
        del_file(org_img_dir_path)
    if not os.path.exists(org_voc_dir_path):
        os.makedirs(org_voc_dir_path)
    else:
        del_file(org_voc_dir_path)
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

def clearfile():
    if os.path.exists(list_train):
        os.remove(list_train)
    if os.path.exists(list_val):
        os.remove(list_val)
    if os.path.exists(list_test):
        os.remove(list_test)# train.cache
    if os.path.exists(train_cache):
        os.remove(train_cache)
    if os.path.exists(val_cache):
        os.remove(val_cache)
    if os.path.exists(img_path):
        shutil.rmtree(img_path)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        
    # if os.path.exists(org_voc_dir_path):
    #     shutil.rmtree(org_voc_dir_path)
    # if os.path.exists(org_img_dir_path):
    #     shutil.rmtree(org_img_dir_path)

def main():#to_yolo_train

    file_train = open(list_train, 'w')
    file_val = open(list_val, 'w')
    file_test = open(list_test, 'w')

    total_txt = os.listdir(label_original_path)
    num_txt = len(total_txt)
    print("Final_num : ",num_txt)
    list_all_txt = range(num_txt)

    num_train = int(num_txt * train_percent)
    num_val = int(num_txt * val_percent)
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

        srcImage = image_original_path + name + jpg
        srcLabel = label_original_path + name + ".txt"

        if i in train:
            # print(name)
            dst_train_Image = train_image_path + name + jpg
            dst_train_Label = train_label_path + name + '.txt'
            shutil.copyfile(srcImage, dst_train_Image)
            shutil.copyfile(srcLabel, dst_train_Label)
            file_train.write(dst_train_Image + '\n')
        elif i in val:
            dst_val_Image = val_image_path + name + jpg
            dst_val_Label = val_label_path + name + '.txt'
            shutil.copyfile(srcImage, dst_val_Image)
            shutil.copyfile(srcLabel, dst_val_Label)
            file_val.write(dst_val_Image + '\n')
        else:
            dst_test_Image = test_image_path + name + jpg
            dst_test_Label = test_label_path + name + '.txt'
            shutil.copyfile(srcImage, dst_test_Image)
            shutil.copyfile(srcLabel, dst_test_Label)
            file_test.write(dst_test_Image + '\n')

    file_train.close()
    file_val.close()
    file_test.close()
    return     print("train_num：{}, val_num：{}, test_num：：{}".format(len(train), len(val), len(val_test) - len(val)))


    # return image_list
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
def convert_annotation(dir_path, output_path, image_path):
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
        if cls not in classes or int(difficult)==1:
            print("difficult == 1")
            continue

        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

def voc2yolotxt():
    cwd = getcwd()
    full_dir_path = cwd + '/' +'yoma_data'+ '/voc/'
    # print(full_dir_path)
    output_path = cwd + '/' +'yoma_data' +'/label/'
    img_path = cwd + '/' +'yoma_data' +'/images/'
    # print(output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    image_paths = getImagesInDir(img_path)
    # print(image_paths)
    list_file = open(full_dir_path + '.txt', 'w')
    #print(list_file)

    for image_path in image_paths:
        list_file.write(image_path + '\n')
        convert_annotation(full_dir_path, output_path, image_path)
    list_file.close()

# augmentations_yoma()
username  = 'yoma'
cwd = getcwd()
input_image_path = cwd + '/' +'yoma_data' +'/input_images/'
input_voc_path = cwd + '/' +'yoma_data' +'/input_voc/'
org_voc_dir_path = cwd + '/' +'yoma_data'+ '/org_voc/'
org_img_dir_path = cwd + '/' +'yoma_data' +'/org_images/'

aug_voc_dir_path = cwd + '/' +'yoma_data'+ '/voc/'
aug_img_dir_path = cwd + '/' +'yoma_data' +'/images/'
# image_paths = getImagesInDir(org_img_dir_path)
# print("all",image_paths)


full_dir_path = cwd + '/' +'yoma_data'+ '/voc/'
# print(full_dir_path)
output_path = cwd + '/' +'yoma_data' +'/label/'
img_path = cwd + '/' +'yoma_data' +'/images/'
# print(output_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)

image_original_path = "./"+username+"_data/images/"
label_original_path = "./"+username+"_data/label/"

cur_path = os.getcwd()
# print(cur_path)

train_image_path = os.path.join(cur_path, "datasets/defect/images/train/")
train_label_path = os.path.join(cur_path, "datasets/defect/labels/train/")
# print(train_image_path)
# 验证集路径
val_image_path = os.path.join(cur_path, "datasets/defect/images/val/")
val_label_path = os.path.join(cur_path, "datasets/defect/labels/val/")

# 测试集路径
test_image_path = os.path.join(cur_path, "datasets/defect/images/test/")
test_label_path = os.path.join(cur_path, "datasets/defect/labels/test/")

# 训练集目录
list_train = os.path.join(cur_path, "datasets/defect/train.txt")
list_val = os.path.join(cur_path, "datasets/defect/val.txt")
train_cache = os.path.join(cur_path, "datasets/defect/train.cache")
val_cache = os.path.join(cur_path, "datasets/defect/val.cache")
list_test = os.path.join(cur_path, "datasets/defect/test.txt")
train_percent = 0.6
val_percent = 0.2
test_percent = 0.2
 ###   voc 2 yolo
classes = ['hole']

clearfile()
mkdir()
count_num2png(input_image_path)
image_paths = getImagesInDir(org_img_dir_path)
# print("input_img_list ",image_paths)
print('augment_num : ',augment_num)
print("____________________Processing____________________")
for image_path in image_paths:
    bboxes = []
    category_ids = []
    classes_list = []

    #print(image_path)
    temp = os.path.splitext(image_path)[0]
    name = os.path.split(temp)[1]
    print("img_name : ",name)

    read_voc(org_voc_dir_path, image_path)

    aug_yoma(image_path,bboxes,name)
# print("Finished augmentations_yoma processing " )
voc2yolotxt()
# print("Finished transfrom processing " )
# print("Finished to_yolo_train processing " )

print("_______________________Done_______________________")
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
print("______________________ERROR_______________________\n")
error_fuc(error_list)
print("______________________ERROR_______________________")
main()
# if error_list == 1 : # 如果error_list 是list 就可以判斷她是哪裡出錯這樣
#     print("!!! error !!!(maybe no have labels)") 
