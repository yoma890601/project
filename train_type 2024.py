##用one_pycharm/point cloud yolo_桌.py 為底改成一個model 不能多個model
## 20231022 改不能只pred 
import os ,time
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
import open3d as o3d

# vram=1024*3
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=vram)])
#   except RuntimeError as e:
#     print(e)


# filename = os.path.basename(__file__)
# filename=filename[17:len(filename)-3]
# print("filename :",filename)
import matplotlib
matplotlib.use('TkAgg')
NUM_POINTS = 2048
NUM_CLASSES = 10
history_list=[]
model_list=[]
preds_list=[]
class_names_list= []
label_list=[]
color = [[0.5,0,0],'b','g','r','y','k','c','m']
npy_path = "./Eclatorq/npy/"
DATA_DIR = "./Eclatorq/type"
save_name_p = "all_obj.npy"
save_name_l = "all_label.npy"

def xyxy2xywh_transfer(temp_xy,imgyx,mode):
    if mode == 'xyxy2xywh':
        xywh = [((temp_xy[0]+temp_xy[2])/2)/imgyx[0],((temp_xy[1]+temp_xy[3])/2)/imgyx[1],abs((temp_xy[0]-temp_xy[2])) /imgyx[0],abs((temp_xy[1]-temp_xy[3])) /imgyx[1]]
        return xywh
    elif mode == 'xywh2xyxy':
        xyxy = [(temp_xy[0]-temp_xy[2]/2)*imgyx[0] ,(temp_xy[1]-temp_xy[3]/2)*imgyx[1] , (temp_xy[0]+temp_xy[2]/2)*imgyx[0] , (temp_xy[1]+temp_xy[3]/2)*imgyx[1]]
        return xyxy

def yoma_pc_aug (train_points):#(1,20,3)
    aug_pc=  []
    get_size = 5
    random_num  = np.sort(np.random.choice(len(train_points), size=get_size, replace=False))
    print(random_num)

    for i in range(get_size): #360度 每1個data 轉10度 >> 360 data
        e_p = train_points[random_num[i]]
        # e_p = np.transpose(e_pc)  # (20,3) t0 (3,20)
        # plt.scatter(e_p[0], e_p[1], color='b') # 原圖 找重心用
        aug_point = []
        deal_point = []
        buf_xmin_org = 0 
        buf_ymin_org = 0 

        for a in range(len(e_p)):
            if np.array(e_p[a]).any():
                if  e_p[a][0] < buf_xmin_org :
                    buf_xmin_org = e_p[a][0]
                if  e_p[a][1] < buf_ymin_org :
                    buf_ymin_org = e_p[a][1]
                #aug_point[a][0]=aug_point[a][0]#-error[0] #  aug_point[a][0] 第a個X
                #aug_point[a][1]=aug_point[a][1]#-error[1]
        print(buf_xmin_org,buf_ymin_org) # ok 
        # print(e_p,'org')
        # print(min(e_p),'min')
        for b in range(len(e_p)):
            if e_p[b] != [0, 0, 0]:
                if buf_xmin_org <= 0:
                    e_p[b][0]= e_p[b][0] - buf_xmin_org
                if buf_ymin_org <= 0:
                    e_p[b][1]= e_p[b][1] - buf_ymin_org
                deal_point.append(e_p[b])
        ### test 
        # for zz in range(len(deal_point)):
        #     plt.scatter(deal_point[zz][0], deal_point[zz][1],color='red') # 沒-重心前
        # print(len(deal_point))
        # plt.show()
        ### test 

        # source = np.asarray(aug_point)
        # try :
        #     pp = o3d.geometry.PointCloud()
        #     pp.points = o3d.utility.Vector3dVector(source)
        #     org_center = pp.get_center()
        # except:
        #     pass
        # print(org_center,'org')
        # plt.scatter(new_point[0], new_point[1],color='red') # 沒-重心前

        # plt.scatter(org_center[0], org_center[1], color='g') # 原圖g的重心

        # for j in range(30) :
        for j in list(range(-30,30,3)):#[30,60]
            # obj_pc = train_points[random_num[i]]
            aa=[]
            for k in range(len(deal_point)):
                if deal_point[k] != [0, 0, 0]:
                    aa.append(deal_point[k])
            test = np.transpose(aa) # (20,3) t0 (3,20)
            theta = j
            rot_matrix = np.array([
            [np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta)), 0],
            [np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta)), 0],
            [0, 0, 1]])
            new_point = np.dot(rot_matrix, test)
            # print(np.shape(new_point))
            # plt.scatter(new_point[0], new_point[1],color='red') # 沒-重心前

            aug_point=np.transpose(new_point)# augment後點 在這
            # print(np.shape(aug_point),"aug_point shape ")

            # source = np.asarray(aug_point)
            # p = o3d.geometry.PointCloud()
            # p.points = o3d.utility.Vector3dVector(source)

            # cc = p.get_center() # aug後重心
            # # print(cc, 'cc')
            # error = cc-org_center
            # print('error',error)
            # plt.scatter(cc[0], cc[1], color='b') # 每個aug的重心

            # print("before",np.shape(aug_point))
            target_num = 20
            buf_xmin  = 0 
            buf_ymin  = 0 


            for a in range(len(aug_point)):
                if np.array(aug_point[a]).any():
                    if  aug_point[a][0] < buf_xmin :
                        buf_xmin = aug_point[a][0]
                    if  aug_point[a][1] < buf_ymin :
                        buf_ymin = aug_point[a][1]

                    #aug_point[a][0]=aug_point[a][0]#-error[0] #  aug_point[a][0] 第a個X
                    #aug_point[a][1]=aug_point[a][1]#-error[1]
            # print(buf_xmin_org,buf_ymin_org) # ok 
            # print(aug_point,'aug')
            for b in range(len(aug_point)):
                if buf_xmin <= 0:
                    aug_point[b][0]= aug_point[b][0] - buf_xmin
                if buf_ymin <= 0:
                    aug_point[b][1]= aug_point[b][1] - buf_ymin


            for zz in range(len(aug_point)):
                plt.scatter(aug_point[zz][0], aug_point[zz][1],color='red') # 沒-重心前
           
            # plt.show()
            for c in range(target_num-len(aug_point)):
                aug_point = list(aug_point)
                aug_point.append([0,0,0])
            # print(np.shape(aug_point))
            # plt.scatter(new_point[0], new_point[1],color='red') # aug-重心後
            
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
        # print(np.shape(train_points))
        # aug_pc = train_points
        plt.show()
        # print(np.shape(train_labels))
        all_train_points.append(aug_pc)
        tttt = train_labels[0]
        print(np.shape(all_train_points))
        print(tttt)
        all_train_labels.append([tttt]*len(aug_pc))

    temp_shape =np.shape(all_train_points)
    print(np.shape(all_train_points))
    # print(np.shape(all_train_labels))

    all_train_points = np.array(all_train_points).reshape((temp_shape[0]*temp_shape[1],temp_shape[2],temp_shape[3]))
    print(all_train_points.shape)

    # print(all_train_points)

    all_train_labels = np.array(all_train_labels).reshape((temp_shape[0]*temp_shape[1]))

    # print(all_train_labels)
    ###
    np.save("./Eclatorq/npy/"+save_name_p, all_train_points)
    np.save("./Eclatorq/npy/"+save_name_l, all_train_labels)
    class_names = np.array(class_names)
    return class_names,all_train_points,all_train_labels   
def parse_dataset(num_points,DATA_DIR):
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

    np.save("./Eclatorq/npy/"+save_name_p, train_points)
    np.save("./Eclatorq/npy/"+save_name_l, train_labels)
    class_names = np.array(class_names)
    return class_names,train_points,train_labels    



def load_dataset(DATA_DIR,use_before=1):
    print("load")
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_names = []
    #return train_name

    data_use_before = 0

    if data_use_before:
        print("use_before")
        folders = glob.glob(os.path.join(DATA_DIR, "*"))
        for i, folder in enumerate(folders):
            temp = os.path.basename(folder)
            print("processing class: {}".format(temp))
            class_names.append(temp)
        train_points = np.load(npy_path+save_name_p, allow_pickle=True)
        train_labels = np.load(npy_path+save_name_l, allow_pickle=True)
        print(train_points.shape)
        print(train_points[1])
            # print(train_points)
        class_names = np.array(class_names)

    elif data_use_before ==0 and os.path.exists(npy_path+save_name_p):
        print("use new and have old version > rename")
        data_time=time.strftime("%Y%m%d_%H%M_",time.localtime(os.path.getmtime (npy_path+save_name_p))) # 日期格式完整>>"%Y-%m-%d %H:%M:%S"
        new_name_p = npy_path+data_time +save_name_p
        new_name_l =npy_path+data_time +save_name_l
        os.rename(npy_path+save_name_p,new_name_p) # "keras.h5" >> "keras.h5"+time 
        os.rename(npy_path+save_name_l,new_name_l) # "keras.h5" >> "keras.h5"+time 
        class_names,train_points,train_labels = augparse_dataset(2048,DATA_DIR) #   augparse_dataset
    else :
        print("else 產生新data")
        # data_time=time.strftime("%Y%m%d_%H%M_",time.localtime(os.path.getmtime (npy_path+save_name_p))) # 日期格式完整>>"%Y-%m-%d %H:%M:%S"
        # new_name_p = npy_path+data_time +save_name_p
        # new_name_l =npy_path+data_time +save_name_l
        # os.rename(npy_path+save_name_p,new_name_p) # "keras.h5" >> "keras.h5"+time 
        # os.rename(npy_path+save_name_l,new_name_l) # "keras.h5" >> "keras.h5"+time 
        class_names,train_points,train_labels = augparse_dataset(2048,DATA_DIR) #   augparse_dataset
        # class_names,train_points,train_labels = parse_dataset(2048,DATA_DIR) #   augparse_dataset

        # folders = glob.glob(os.path.join(DATA_DIR, "*"))
        # for i, folder in enumerate(folders):
        #     temp = os.path.basename(folder)
        #     print("processing class: {}".format(temp))
        #     class_names.append(temp)
        #     train_points = np.load(save_name_p, allow_pickle=True)
        #     train_labels = np.load(save_name_l, allow_pickle=True)
        #     # print(train_points)
        # class_names = np.array(class_names)
    return class_names,train_points,train_labels    
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
def Kfold_ramdom(x,y,splits,random_state,seed):
    kf = KFold(n_splits=splits, random_state=random_state, shuffle=True)
    for train_index, test_index in kf.split(x):
        x_train, x_test = np.array(x)[train_index], np.array(x)[test_index] # 20231022 修正kfold 會報錯問題
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
    index = [i for i in range(len(x_train))]
    np.random.seed(seed)
    np.random.shuffle(index)
    x_train = x_train[index]
    y_train = y_train[index]
    return x_train,x_test,y_train,y_test
def plot_confusion_matrix(label_list, preds_list, classes, normalize,title):
    for j in range (len(label_list)):
        y_true = label_list[j] 
        y_pred= preds_list[j]
        # print(classes)
        temp = classes[j]
        # print(classes)
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_true, y_pred)#
        #要判斷是否跟原本一樣 如果不是 就不顯示  不然epoch太少 label 跟 class 布一樣
        if np.shape(cm)[0]== len(temp) or np.shape(cm)[1] == len(temp):
            # Only use the labels that appear in the data
            #temp = classes[unique_labels(y_true,y_pred)]
            #print(classes)
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                #print("Normalized confusion matrix")
            else:
                pass
                #print('Confusion matrix, without normalization')

            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            # We want to show all ticks...
            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   # ... and label them with the respective list entries
                   xticklabels=temp, yticklabels=temp,
                   title=title[j],
                   ylabel='True label',
                   xlabel='Predicted label')

            ax.set_ylim(len(temp)-0.5, -0.5)

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")

            # Loop over data dimensions and create text annotations.
            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], fmt),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            fig.tight_layout()
        else:
            print("model不完整 有class沒被辨識出來")
def plot_history(history_list):
    save_history=[]
    for i in history_list[0].history:
        save_history.append(i)
    print(save_history)
    for i in range (len(save_history)):
        fig, ax = plt.subplots()
        for j in range (len(history_list)):
            ax.plot(history_list[j].history[save_history[i]], color=color[j])
        ax.legend(model_list, loc='upper left')
        ax.set(title="Model "+save_history[i],ylabel=save_history[i],xlabel= 'Epoch')
def model0(x_train,y_train,x_test,y_test,class_names,BATCH_SIZE,epochs,model_weights_name):
    inputs = tf.keras.Input(shape=(20, 3))
    ##  orginal model
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
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
    # model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["sparse_categorical_accuracy"], )
    history = model.fit(x_train, y_train,validation_data = (x_test,y_test),batch_size=BATCH_SIZE, epochs=epochs, verbose=1)
    
    if os.path.exists(model_weights_name): # 20231022 新增訓練後保留上一個權重為訓練時的名字 新的為keras.h5 辨識那邊會預設選用
        # w = os.path.getmtime ("keras.h5") # 修改時間 getmtime  ##建立時間 getctime 存取時間 getatime 
        data_time=time.strftime("%Y%m%d_%H%M_",time.localtime(os.path.getmtime ("keras.h5"))) # 日期格式完整>>"%Y-%m-%d %H:%M:%S"
        new_name = str(data_time)+model_weights_name
        print("last file rename to : ",new_name)
        try:
            os.rename(model_weights_name,new_name) # "keras.h5" >> "keras.h5"+time
        except Exception as e:
            print(e)
        model.save_weights(model_weights_name)

    preds = model.predict(x_test)
    label = y_test
    return history,preds,label

class test_train():
    def __init__(self,show = False,epochs = 100):
        #self.DATA_DIR = ["C:/Users/ccu/Downloads/ModelNet10/ModelNet10","C:/Users/ccu/Downloads/ModelNet10/ModelNet10"]
        # save_tf = False
        self.fit_tf = True # True  False
        self.epochs = epochs
        self.BATCH_SIZE_lsit = 16
        self.splits = 20
        self.model_i = 0 # 指 上面的model0 如果新增model1  改成1就會train mdoel1
        self.show_tf = show
        #fit_list = [x_train,x_train]
    def run(self):

        history = "history"+str(self.model_i)
        model_save = "model"+str(self.model_i)
        # print(model_save)
        model_list.append(model_save) 
        model = eval("model"+str(self.model_i))

        model_weights_name= "keras.h5"
        class_names,train_points,train_labels = load_dataset(DATA_DIR)
######
        x_train,x_test,y_train,y_test = Kfold_ramdom(train_points,train_labels,self.splits,666,87)
        # print(class_names)
        # fit
        history,preds,label =model(x_train,y_train,x_test,y_test,class_names,self.BATCH_SIZE_lsit,self.epochs,model_weights_name) #BATCH_SIZE,self.epochs

        history_list.append(history)

        if self.show_tf ==True:
            buf_pr = preds##給後面顯示用
            preds = np.argmax(preds, -1) ## 分類
            label_list.append(label)
            preds_list.append(preds)
            class_names_list.append(class_names)

            plot_confusion_matrix(label_list, preds_list, classes=class_names_list, normalize=False,title=model_list)
            plot_history(history_list)
            plt.show()
        elif self.show_tf ==False:
            preds = np.argmax(preds, -1) ## 分類
            label_list.append(label)
            preds_list.append(preds)
            class_names_list.append(class_names)
            plot_confusion_matrix(label_list, preds_list, classes=class_names_list, normalize=False,title=model_list)
            plt.show()
            print("show_tf : ",self.show_tf )
            pass
######
# DATA_DIR = ["./Eclatorq/type","./Eclatorq/type"]  # model 0 用10種 model 1 用5種
# #DATA_DIR = ["C:/Users/ccu/Downloads/ModelNet10/ModelNet10","C:/Users/ccu/Downloads/ModelNet10/ModelNet10"]
# save_tf = False
# fit_tf = True # True  False
# epochs = 100
# BATCH_SIZE_lsit = [16,16,16]
# model_num= 1
# splits = 20
# #fit_list = [x_train,x_train]
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
#         history,preds,label=  (x_train,y_train,x_test,y_test,class_names,BATCH_SIZE_lsit[i],epochs,model_weights_name,fit_tf) #BATCH_SIZE,epochs
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
