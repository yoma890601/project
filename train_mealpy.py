# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import os
import glob
import time
import datetime
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

# vram = 1024 * 3
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_virtual_device_configuration(gpus[0], [
#             tf.config.experimental.VirtualDeviceConfiguration(memory_limit=vram)])
#     except RuntimeError as e:
#         print(e)
import sys

history_list = []
class_names_list = []
model_list = []
preds_list = []
label_list = []
sys.path.append('../')
# End of fix
# DATA_DIR ="C:/Users/karta/Downloads/ModelNet10/npp"
DATA_DIR ="./Eclatorq/sop/type"

color = [[0.5, 0, 0], 'b', 'g', 'r', 'y', 'k', 'c', 'm', 'gray', 'lightgray']
from niapy.algorithms.basic import ParticleSwarmAlgorithm
from niapy.task import Task
import numpy as np
from niapy.problems.problem import Problem


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
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_names = []

    # return train_name
    save_name_p ="all_obj.npy" #"5_800_all_obj.npy" 
    save_name_l ="all_label.npy" #"5_800_all_label.npy" 
    base_path = "./Eclatorq/npy"

    print("DATA_DIR : ", DATA_DIR)
    print("save : ", save_name_p)
    folders = glob.glob(os.path.join(DATA_DIR, "*"))

    for i, folder in enumerate(folders):
        temp = os.path.basename(folder)
        print("processing class: {}".format(temp))
        class_names.append(temp)
        train_points = np.load(base_path+'/'+save_name_p, allow_pickle=True)
        train_labels = np.load(base_path+'/'+save_name_l, allow_pickle=True)
    class_names = np.array(class_names)
    return class_names, train_points, train_labels


def plot_confusion_matrix(label_list, preds_list,model_num, classes, normalize):
    for j in range(model_num):
        y_true = label_list[j]
        y_pred = preds_list[j]
        temp = classes[j]
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        # temp = classes[unique_labels(y_true,y_pred)]
        # print(classes)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            # print("Normalized confusion matrix")
        else:
            pass
            # print('Confusion matrix, without normalization')

        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=temp, yticklabels=temp,
               title=model_list[j],
               ylabel='True label',
               xlabel='Predicted label')

        ax.set_ylim(len(temp) - 0.5, -0.5)

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


def plot_history(history_list,model_num):
    save_history = []
    for i in history_list[0].history:
        save_history.append(i)
    print(save_history)
    for i in range(len(save_history)):
        fig, ax = plt.subplots()
        for j in range(int(model_num-1)):
            ax.plot(history_list[j].history[save_history[i]], color=color[j])
        if save_history[i] =='loss':
            ax.set(title='Loss', ylabel='Loss', xlabel='Epoch')
            ax.legend(model_list[1:len(model_list)], loc='upper right')
        else:
            ax.legend(model_list[1:len(model_list)], loc='lower right')
            ax.set(title='Accuracy', ylabel='Accuracy', xlabel='Epoch')
        # ax.legend(['1','2','3','4','5','6','7','8','9','10'], loc='upper left')


ttime = datetime.datetime.now()
ttime = str(ttime)
print(ttime)

algo = ParticleSwarmAlgorithm(population_size=100, min_velocity=-4.0, max_velocity=4.0)
# Neurons_list=[16,32,64,128,256,512]
# Dropout_list=[0,0.1,0.2,0.3,0.4,0.5]
Neurons_list = [16, 32, 64, 128, 256]
Dropout_list = [0, 0.1, 0.2, 0.3, 0.4]
learnnig_curve = []
all_learnnig_curve = []
def yomatnet(inputs, num_features):
    # Initalise bias as the indentity matrix
    bias = tf.keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, num_features)
    x = conv_bn(x, num_features*2)
    x = conv_bn(x, num_features*16)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, num_features*8)
    x = dense_bn(x, num_features*4)
    x = layers.Dense(num_features * num_features, kernel_initializer="zeros", bias_initializer=bias,
                     activity_regularizer=reg, )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])
def model_or(x_train, y_train, x_test, y_test, class_names, batch_size, parameter,kf_name,save_name):
    print("or")
    Dropout_0 = 0.3
    Neurons_1 = int(parameter[0])  # Neurons_list[int(parameter[1])] #32
    Neurons_2 = 32
    Neurons_3 = Neurons_2 * 2
    Neurons_4 = int(parameter[1])  # 512
    Neurons_5 = int(parameter[2])  # 256
    Neurons_6 = int(parameter[3])  # 128

    # print(Dropout_0,Neurons_1,Neurons_2,Neurons_3,Neurons_4,Neurons_5,Neurons_6,Neurons_7)


    inputs = tf.keras.Input(shape=(20, 3))
    ##  orginal model
    x = tnet(inputs, 3)
    x = conv_bn(x, 32)
    x = conv_bn(x, 32)
    x = tnet(x, 32)
    x = conv_bn(x, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = layers.Dropout(0.3)(x)
    x = dense_bn(x, 128)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
    # model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
                  metrics=["sparse_categorical_accuracy"], )
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=0)

    preds = model.predict(x_test)
    label = y_test
    temp = str(save_name)[len(save_name) - 8:len(save_name)]+'_'+str(kf_name)
    # print(temp)
    if kf_name =='meal':
        f_n = './Eclatorq/train_meal_h5/'
    else :
        f_n = './Eclatorq/train_nia_h5/'

    model_weights_name = f_n+'pcm_or_' + temp + '.h5'
    model_name = 'Original'
    model_list.append(model_name)
    model.save_weights(model_weights_name)
    return history, preds, label
def model_touse(x_train, y_train, x_test, y_test, class_names, model_weights_name, parameter, times):
    print("model_touse")
    Dropout_0 = 0.3
    Neurons_1 = int(parameter[0])  # Neurons_list[int(parameter[1])] #32
    Neurons_2 = 32
    Neurons_3 = Neurons_2 * 2
    Neurons_4 = int(parameter[1])  # 512
    Neurons_5 = int(parameter[2])  # 256
    Neurons_6 = int(parameter[3])  # 128

    # print(Dropout_0,Neurons_1,Neurons_2,Neurons_3,Neurons_4,Neurons_5,Neurons_6,Neurons_7)
    # print(" tnet :","3",'\n',"conv_bn :",Neurons_1,'\n',"conv_bn :",Neurons_2,'\n',"tnet :",Neurons_2,'\n',"conv_bn :",Neurons_2,'\n'
    # ,"conv_bn :",Neurons_3,'\n',"conv_bn :",Neurons_4,'\n',"GlobalMaxPooling1D",'\n',"dense_bn :",Neurons_5,'\n'
    # ,"Dropout :",Dropout_0,'\n',"dense_bn :",Neurons_6,'\n',"Dropout :",Dropout_0,'\n',"output_Dense :",len(class_names),'\n')

    inputs = tf.keras.Input(shape=(20, 3))
    ##  orginal model
    x = tnet(inputs, 3)
    x = conv_bn(x, Neurons_1)
    x = conv_bn(x, Neurons_2)  # 32
    x = yomatnet(x, Neurons_2)  # 32
    x = conv_bn(x, Neurons_2)  # 32
    x = conv_bn(x, Neurons_3)  # 64
    x = conv_bn(x, Neurons_4)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, Neurons_5)
    x = layers.Dropout(Dropout_0)(x)  # 0.3
    x = dense_bn(x, Neurons_6)
    x = layers.Dropout(Dropout_0)(x)  # 0.3

    outputs = layers.Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
    # model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
                  metrics=["sparse_categorical_accuracy"], )
    # model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=1)
    model.load_weights(model_weights_name + '.h5')
    label = y_test
    preds = model.predict(x_test)
    return preds, label
def model(x_train, y_train, x_test, y_test, class_names, batch_size, parameter,kf_name,save_name):
    print("fittt")
    Dropout_0 = 0.3
    Neurons_1 = int(parameter[0])  # Neurons_list[int(parameter[1])] #32
    Neurons_2 = 32
    Neurons_3 = Neurons_2 * 2
    Neurons_4 = int(parameter[1])  # 512
    Neurons_5 = int(parameter[2])  # 256
    Neurons_6 = int(parameter[3])  # 128

    # print(Dropout_0,Neurons_1,Neurons_2,Neurons_3,Neurons_4,Neurons_5,Neurons_6,Neurons_7)
    print(" tnet :", "3", '\n', "conv_bn :", Neurons_1, '\n', "conv_bn :", Neurons_2, '\n', "tnet :", Neurons_2, '\n',
          "conv_bn :", Neurons_2, '\n'
          , "conv_bn :", Neurons_3, '\n', "conv_bn :", Neurons_4, '\n', "GlobalMaxPooling1D", '\n', "dense_bn :",
          Neurons_5, '\n'
          , "Dropout :", Dropout_0, '\n', "dense_bn :", Neurons_6, '\n', "Dropout :", Dropout_0, '\n', "output_Dense :",
          len(class_names), '\n')

    inputs = tf.keras.Input(shape=(20, 3))
    ##  orginal model
    x = tnet(inputs, 3)
    x = conv_bn(x, Neurons_1)
    x = conv_bn(x, Neurons_2)  # 32
    x = yomatnet(x, Neurons_2)  # 32
    x = conv_bn(x, Neurons_2)  # 32
    x = conv_bn(x, Neurons_3)  # 64
    x = conv_bn(x, Neurons_4)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, Neurons_5)
    x = layers.Dropout(Dropout_0)(x)  # 0.3
    x = dense_bn(x, Neurons_6)
    x = layers.Dropout(Dropout_0)(x)  # 0.3

    outputs = layers.Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
    # model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
                  metrics=["sparse_categorical_accuracy"], )
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=0)

    preds = model.predict(x_test)
    label = y_test
    temp = str(save_name)[len(save_name) - 8:len(save_name)]+'_'+str(kf_name)
    # print(temp)
    if kf_name =='kf_nia':
        f_n = './Eclatorq/train_nia_h5/'
    else :
        f_n = './Eclatorq/train_meal_h5/'

    model_weights_name = f_n+'pcm_' + temp + '.h5'
    model_name = str(kf_name)
    model_list.append(model_name)
    model.save_weights(model_weights_name)
    return history, preds, label

fit = 1

# f_n = "meal_05082127"
# print(f_n[0:6])
# if f_n[0:6]=='kf_nia':
#     save_name = './nia_data/' + f_n  # "kf_nia_07170910"
#     train_type = 'nia_data'
# elif f_n[0:5]=='meal_':
#     save_name = './meal_data/'+f_n  # "kf_nia_07170910"
#     train_type = 'meal_data'

# print(save_name)

# all_temp_best_acc_list = np.load(str(save_name) + 'acc.npy', allow_pickle=True)


# all_best = np.load(str(save_name)+'all.npy', allow_pickle=True)
# if train_type =='nia_data':
#     all_best_fit = np.load(str(save_name) + 'fit.npy', allow_pickle=True)

#     temp_best = 0
#     kf_best_index = []
#     for i in range(len(all_temp_best_acc_list)):
#         if max(all_temp_best_acc_list[i]) > temp_best:
#             temp_best = max(all_temp_best_acc_list[i])
#     for i in range(len(all_temp_best_acc_list)):
#         if max(all_temp_best_acc_list[i]) == temp_best:
#             print("在第", i + 1, "個fold 中共", len(all_temp_best_acc_list[i]), "個 evals")
#             kf_best_index.append(i + 1)
#             print(all_temp_best_acc_list[i])
#             print("parameter : ")
#             parameter = all_best_fit[i]
#             print('[' + str(all_best_fit[i][0]) + ',' + str(all_best_fit[i][1]) + ',' + str(all_best_fit[i][2]) + ',' + str(
#                 all_best_fit[i][3]) + ']')
#             print("分數最高為 ", temp_best)
#         print("")
#         model_num = len(kf_best_index)
#         # print('model_num = ',model_num)
# elif train_type =='meal_data':
#     all_buf_fit_list = np.load(str(save_name) + 'fit.npy', allow_pickle=True)
#     # np.asanyarray()
#     all_buf_acc_list = np.load(str(save_name) + 'bufacc.npy', allow_pickle=True)
#     # all_buf_fit_list = np.load(str(save_name) + 'buffit.npy', allow_pickle=True)
#     model_name = np.load(str(save_name) + 'model_name.npy', allow_pickle=True)
#     tempp1 = []
#     tempp = []
#     # print('all_temp_best_acc_list \n',all_temp_best_acc_list[0])
#     print('all_temp_best_acc_list \n',model_name)

#     # for i in range(len(all_buf_fit_list)):
#     #     temp =  np.asanyarray(all_buf_fit_list[i])
#     #     tempp.append(temp)
#     #     temp1 =  np.asanyarray(all_buf_acc_list[i])
#     #     tempp1.append(temp1)
#     # all_buf_fit_list = tempp
#     # all_buf_acc_list = tempp1


#     model_num = len(model_name)

#     # 上面是因為有kf 所以每個fold不一樣
#     # print('all_temp_best_acc_list \n',all_temp_best_acc_list)
#     best_fit = []
#     # print('all_buf_acc_list \n',all_buf_acc_list)
#     print(model_name)
#     fig, ax = plt.subplots()

#     for i in range (len(all_temp_best_acc_list)):
#         ax.plot(all_temp_best_acc_list[i], color=color[i])
#         ax.set(title='lc_'+" meal _"+str(ttime[5:7]+ttime[8:10]+ttime[11:13]+ttime[14:16])+'_acc_') # str(np.array(all_temp_best_acc_list).max())
#         ax.legend(model_name, loc='lower right')
#     # plt.show()
#     for i in range(model_num):
#         algo_best = np.max(all_buf_acc_list[i])#
#         all_buf_acc_list_index = np.where(all_buf_acc_list[i] == algo_best)
#         # print(all_buf_fit_list[i][all_buf_acc_list_index][0])
#         print(all_buf_fit_list[i][all_buf_acc_list_index][0])
#         best_fit.append(all_buf_fit_list[i][all_buf_acc_list_index][0])
#         parameter = best_fit[i]  ####下面要改 可以2跑兩個演算法的    best_fit[0]#all_best_fit
#         print()
#         print('model_name : ', model_name[i])

#         print("parameter : ")

#         print('[' + str(parameter[0]) + ',' + str(parameter[1]) + ',' + str(parameter[2]) + ',' + str(
#             parameter[3]) + ']')
#         print("分數最高為 ", algo_best)
#     # print(best_fit)
#     # print('model_name : \n',model_name)

#     print('meal_fit_list : \n',best_fit)

#     # print('meal_best_fit : \n',all_best_fit)# 改完應該要有兩個  先確定數據都沒錯
#                                             # 理論上 換parameter 就能train 不同的 然後加上 model_name 後綴
#                                             # 畫出各演算法差異
#     #
#     # print("parameter : ")
#     #
#     # parameter =best_fit[0] ####下面要改 可以2跑兩個演算法的    best_fit[0]#all_best_fit
#     # print('[' + str(parameter[0]) + ',' + str(parameter[1]) + ',' + str(parameter[2]) + ',' + str(
#     #     parameter[3]) + ']')
#     # temp_best = all_temp_best_acc_list.max()
#     # print("分數最高為 ", temp_best)

# class_names, train_points, train_labels = load_dataset(DATA_DIR)
# # x_train,x_test,y_train,y_test = Kfold_ramdom(train_points,train_labels,20,666,87)
# kf_times = 0
# temp_best_acc = 0
# temp_best_acc_list = []
# val_tf = False  # False True
# kf = KFold(n_splits=10, random_state=666, shuffle=True)

# if fit == 1 :
#     if train_type =='nia_data':
#         for train_index, test_index in kf.split(train_points):
#             x_val = []
#             y_val = []
#             kf_times += 1
#             np.random.seed(1688)
#             # print("KF",kf_times)
#             tf.compat.v1.reset_default_graph()
#             x_train, x_test = train_points[train_index], train_points[test_index]
#             y_train, y_test = train_labels[train_index], train_labels[test_index]
#             # print(len(x_train))
#             x_train = list(x_train)
#             y_train = list(y_train)

#             pop_index_list = np.random.choice(len(x_train), int(len(x_train) / 10), replace=False)
#             if val_tf == False:
#                 # print("pass")
#                 pass
#             elif val_tf == True:
#                 for i in pop_index_list:
#                     if i < len(x_train):
#                         temp_list_pop = x_train.pop(i)
#                         x_val.append(temp_list_pop)
#                         temp_list_pop = y_train.pop(i)
#                         y_val.append(temp_list_pop)
#             # print(len(x_val))
#             x_train = np.array(x_train)
#             y_train = np.array(y_train)
#             x_val = np.array(x_val)
#             y_val = np.array(y_val)
#             index = [i for i in range(len(x_train))]
#             np.random.seed(1688)
#             np.random.shuffle(index)
#             x_train = x_train[index]
#             y_train = y_train[index]
#             # 在KF第 1 fold 誤差最小為 : 0.1440217391304348 神經元為 [ 12.06658145 105.81916911 153.28292629 213.04487111]
#             if kf_times in kf_best_index:
#                 parameter = all_best_fit[kf_best_index.index(kf_times)]  # [12.06658145 ,105.81916911, 153.28292629 ,213.04487111]
#                 print(parameter)
#                 kf_name = kf_times
#                 use = False
#                 if val_tf == False:  ##家這個出錯
#                     if use == False:
#                         print("是否用有val : ", val_tf)

#                         history, preds, label = model_or(x_train, y_train, x_test, y_test, class_names, 32, parameter,kf_name)  # BATCH_SIZE,epochs
#                         print (model_num)
#                         model_num =model_num+1
#                         print(model_num)

#                         preds = np.argmax(preds, -1)  ## 分類
#                         label_list.append(label)
#                         preds_list.append(preds)
#                         # history_list.append(history)
#                         class_names_list.append(class_names)

#                         same = 0
#                         for i in range(len(preds)):
#                             if preds[i] == label[i]:
#                                 same += 1
#                         acc = same / len(preds)
#                         print(acc, "正確率")
#                         if acc > temp_best_acc:
#                             temp_best_acc = acc
#                         temp_best_acc_list.append(temp_best_acc)

#                         point = history.history['sparse_categorical_accuracy']
#                         # print(point)
#                         model_max_acc = max(point)
#                         model_last_acc = point[-1]
#                         print(model_max_acc, "or_model_max_acc")
#                         print(model_last_acc, "or_model_last_acc")

#                         history, preds, label = model(x_train, y_train, x_test, y_test, class_names, 32, parameter,kf_name)  # BATCH_SIZE,epochs
#                         preds = np.argmax(preds, -1)  ## 分類
#                         label_list.append(label)
#                         preds_list.append(preds)
#                         history_list.append(history)
#                         class_names_list.append(class_names)

#                         same = 0
#                         for i in range(len(preds)):
#                             if preds[i] == label[i]:
#                                 same += 1
#                         acc = same / len(preds)
#                         print(acc, "正確率")
#                         if acc > temp_best_acc:
#                             temp_best_acc = acc
#                         temp_best_acc_list.append(temp_best_acc)

#                         point = history.history['sparse_categorical_accuracy']
#                         # print(point)
#                         model_max_acc = max(point)
#                         model_last_acc = point[-1]
#                         print(model_max_acc, "model_max_acc")
#                         print(model_last_acc, "model_last_acc")
#                         plt.rcParams['font.size'] = 14
#                         # plot_confusion_matrix(label_list, preds_list, classes=class_names_list, normalize=False,title="confusion_matrix")
#                         # plot_confusion_matrix(label_list, preds_list, classes=class_names_list, normalize=False)

#                         # plot_history(history_list)
#                         # plt.show()
#                     elif use == True:
#                         print(use)
#                         preds, label = model_touse(x_train, y_train, x_test, y_test, class_names, 'pcm_predict_07170910',
#                                                    parameter, i)  # BATCH_SIZE,epochs

#                         preds = np.argmax(preds, -1)  ## 分類
#                         # label_list.append(label)
#                         preds_list.append(preds)
#                         # history_list.append(history)
#                         class_names_list.append(class_names)

#                         same = 0
#                         for i in range(len(preds)):
#                             if preds[i] == label[i]:
#                                 same += 1
#                         acc = same / len(preds)
#                         print(acc, "正確率")
#                         if acc > temp_best_acc:
#                             temp_best_acc = acc
#                         temp_best_acc_list.append(temp_best_acc)

#                         # point = history.history['sparse_categorical_accuracy']
#                         # print(point)
#                         # model_max_acc = max(point)
#                         # model_last_acc = point[-1]
#                         # print(model_max_acc,"model_max_acc")
#                         # print(model_last_acc,"model_last_acc")
#                         pass

#                     # print(model_list[i])

#                 elif val_tf == True:
#                     pass
#                     # print("是否用有val : ",val_tf)
#                     # history,preds,label= model(x_train,y_train,x_val,y_val,class_names,32,parameter,i) #BATCH_SIZE,epochs
#     elif train_type =='meal_data':
#         x_val = []
#         y_val = []
#         kf_name = 'meal'
#         for train_index, test_index in kf.split(train_points):
#             np.random.seed(1688)
#             tf.compat.v1.reset_default_graph()
#             x_train, x_test = train_points[train_index], train_points[test_index]
#             y_train, y_test = train_labels[train_index], train_labels[test_index]
#         # x_train = train_points
#         # y_train = train_labels

#         print(len(x_train))

#         x_train = list(x_train)
#         y_train = list(y_train)

#         np.random.seed(1688)
#         pop_index_list = np.random.choice(len(x_train), int(len(x_train) / 10), replace=False)
#         # for i in range (len(x_train)-int(len(x_train)/10)):
#         #     if i%10 ==0:
#         #         list_pop=x_train.pop(i)
#         #         x_val.append(list_pop)
#         #         list_pop=y_train.pop(i)
#         #         y_val.append(list_pop)
#         print(len(pop_index_list), "pop val ")
#         for i in pop_index_list:
#             if i < len(x_train):
#                 temp_list_pop = x_train.pop(i)
#                 x_val.append(temp_list_pop)
#                 temp_list_pop = y_train.pop(i)
#                 y_val.append(temp_list_pop)
#         print(len(x_val))
#         x_train = np.array(x_train)
#         y_train = np.array(y_train)
#         x_val = np.array(x_val)
#         y_val = np.array(y_val)

#         index = [i for i in range(len(x_train))]
#         np.random.seed(1688)
#         np.random.shuffle(index)
#         x_train = x_train[index]
#         y_train = y_train[index]

#         history, preds, label = model_or(x_train, y_train, x_val, y_val, class_names, 32, parameter,kf_name)  # BATCH_SIZE,epochs
#         print(model_num)
#         model_num = model_num + 1
#         print(model_num)

#         preds = np.argmax(preds, -1)  ## 分類
#         label_list.append(label)
#         preds_list.append(preds)
#         history_list.append(history)
#         class_names_list.append(class_names)

#         same = 0
#         for i in range(len(preds)):
#             if preds[i] == label[i]:
#                 same += 1
#         or_acc = same / len(preds)
#         print(or_acc, "正確率")
#         if or_acc > temp_best_acc:
#             temp_best_acc = or_acc
#         temp_best_acc_list.append(temp_best_acc)

#         point = history.history['sparse_categorical_accuracy']
#         # print(point)
#         model_max_acc = max(point)
#         model_last_acc = point[-1]
#         print(model_max_acc, "or_model_max_acc")
#         print(model_last_acc, "or_model_last_acc")
#         print(or_acc)

#         if model_num >1 and or_acc <0.99:
#             for i in range (model_num-1):
#                 print(i)
#                 parameter = best_fit[i]
#                 kf_name = model_name[i]
#                 history, preds, label = model(x_train, y_train, x_val, y_val, class_names, 32, parameter,
#                                               kf_name)  # BATCH_SIZE,epochs
#                 preds = np.argmax(preds, -1)  ## 分類
#                 label_list.append(label)
#                 preds_list.append(preds)
#                 history_list.append(history)
#                 class_names_list.append(class_names)

#                 same = 0
#                 for i in range(len(preds)):
#                     if preds[i] == label[i]:
#                         same += 1
#                 acc = same / len(preds)
#                 print(acc, "正確率")
#                 if acc > temp_best_acc:
#                     temp_best_acc = acc
#                 temp_best_acc_list.append(temp_best_acc)

#                 point = history.history['sparse_categorical_accuracy']
#                 # print(point)
#                 model_max_acc = max(point)
#                 model_last_acc = point[-1]
#                 print(model_max_acc, "model_max_acc")
#                 print(model_last_acc, "model_last_acc")
#         elif model_num ==1:
#             history, preds, label = model(x_train, y_train, x_val, y_val, class_names, 32, parameter,
#                                           kf_name)  # BATCH_SIZE,epochs
#             preds = np.argmax(preds, -1)  ## 分類
#             label_list.append(label)
#             preds_list.append(preds)
#             history_list.append(history)
#             class_names_list.append(class_names)

#             same = 0
#             for i in range(len(preds)):
#                 if preds[i] == label[i]:
#                     same += 1
#             acc = same / len(preds)
#             print(acc, "正確率")
#             if acc > temp_best_acc:
#                 temp_best_acc = acc
#             temp_best_acc_list.append(temp_best_acc)

#             point = history.history['sparse_categorical_accuracy']
#             # print(point)
#             model_max_acc = max(point)
#             model_last_acc = point[-1]
#             print(model_max_acc, "model_max_acc")
#             print(model_last_acc, "model_last_acc")
#         # plt.rcParams['font.size'] = 20
#         fig, ax = plt.subplots()
#         plt.rcParams['font.size'] = 20

#         for i in range (len(all_temp_best_acc_list)):
#             ax.plot(all_temp_best_acc_list[i], color=color[i])
#             # print(model_name)
#             list(model_name).insert(0,['or'])
#             # print(model_name)
#             ax.legend(model_name, loc='lower right')
#             print(max(all_temp_best_acc_list[i]))
#             ax.set(title='learning curve')
#     plt.rcParams['font.size'] = 16
#     plot_history(history_list)
#     plt.rcParams['font.size'] = 20

#     plot_confusion_matrix(label_list, preds_list, classes=class_names_list, normalize=False)

#     plt.show()
# elif fit == 0 :
#     print('pass')



class train_meal():
    def __init__(self,f_n = '',show =0):
        fit = 1
        self.f_n = f_n
        self.show = show
        print(f_n[0:6])
        self.save_name =''
        if self.f_n[0:6]=='kf_nia':
            self.save_name = './Eclatorq/nia_data/' + self.f_n  # "kf_nia_07170910"
            self.train_type = 'nia_data'
        elif self.f_n[0:5]=='meal_':
            self.save_name = './Eclatorq/meal_data/'+self.f_n  # "kf_nia_07170910"
            self.train_type = 'meal_data'


    def run(self):
        all_temp_best_acc_list = np.load(str(self.save_name) + 'acc.npy', allow_pickle=True)
        self.train_or = 0
        if self.train_type =='nia_data':
            all_best_fit = np.load(str(self.save_name) + 'fit.npy', allow_pickle=True)

            temp_best = 0
            kf_best_index = []
            for i in range(len(all_temp_best_acc_list)):
                if max(all_temp_best_acc_list[i]) > temp_best:
                    temp_best = max(all_temp_best_acc_list[i])
            for i in range(len(all_temp_best_acc_list)):
                if max(all_temp_best_acc_list[i]) == temp_best:
                    print("在第", i + 1, "個fold 中共", len(all_temp_best_acc_list[i]), "個 evals")
                    kf_best_index.append(i + 1)
                    print(all_temp_best_acc_list[i])
                    print("parameter : ")
                    parameter = all_best_fit[i]
                    print('[' + str(all_best_fit[i][0]) + ',' + str(all_best_fit[i][1]) + ',' + str(all_best_fit[i][2]) + ',' + str(
                        all_best_fit[i][3]) + ']')
                    print("分數最高為 ", temp_best)
                print("")
                model_num = len(kf_best_index)
                # print('model_num = ',model_num)
        elif self.train_type =='meal_data':
            all_buf_fit_list = np.load(str(self.save_name) + 'fit.npy', allow_pickle=True)
            # np.asanyarray()
            all_buf_acc_list = np.load(str(self.save_name) + 'bufacc.npy', allow_pickle=True)
            # all_buf_fit_list = np.load(str(self.save_name) + 'buffit.npy', allow_pickle=True)
            model_name = np.load(str(self.save_name) + 'model_name.npy', allow_pickle=True)
            tempp1 = []
            tempp = []
            # print('all_temp_best_acc_list \n',all_temp_best_acc_list[0])
            print('all_temp_best_acc_list \n',model_name)

            # for i in range(len(all_buf_fit_list)):
            #     temp =  np.asanyarray(all_buf_fit_list[i])
            #     tempp.append(temp)
            #     temp1 =  np.asanyarray(all_buf_acc_list[i])
            #     tempp1.append(temp1)
            # all_buf_fit_list = tempp
            # all_buf_acc_list = tempp1


            model_num = len(model_name)

            # 上面是因為有kf 所以每個fold不一樣
            # print('all_temp_best_acc_list \n',all_temp_best_acc_list)
            best_fit = []
            # print('all_buf_acc_list \n',all_buf_acc_list)
            print(model_name)
            fig, ax = plt.subplots()

            for i in range (len(all_temp_best_acc_list)):
                ax.plot(all_temp_best_acc_list[i], color=color[i])
                ax.set(title='lc_'+" meal _"+str(ttime[5:7]+ttime[8:10]+ttime[11:13]+ttime[14:16])+'_acc_') # str(np.array(all_temp_best_acc_list).max())
                ax.legend(model_name, loc='lower right')
            # plt.show()
            for i in range(model_num):
                algo_best = np.max(all_buf_acc_list[i])#
                all_buf_acc_list_index = np.where(all_buf_acc_list[i] == algo_best)
                # print(all_buf_fit_list[i][all_buf_acc_list_index][0])
                best_fit.append(all_buf_fit_list[i][all_buf_acc_list_index][0])
                parameter = best_fit[i]  ####下面要改 可以2跑兩個演算法的    best_fit[0]#all_best_fit
                print()
                print('model_name : ', model_name[i])

                print("parameter : ")

                print('[' + str(parameter[0]) + ',' + str(parameter[1]) + ',' + str(parameter[2]) + ',' + str(
                    parameter[3]) + ']')
                print("分數最高為 ", algo_best)
            # print(best_fit)
            # print('model_name : \n',model_name)

            print('meal_fit_list : \n',best_fit)

            # print('meal_best_fit : \n',all_best_fit)# 改完應該要有兩個  先確定數據都沒錯
                                                    # 理論上 換parameter 就能train 不同的 然後加上 model_name 後綴
                                                    # 畫出各演算法差異
            #
            # print("parameter : ")
            #
            # parameter =best_fit[0] ####下面要改 可以2跑兩個演算法的    best_fit[0]#all_best_fit
            # print('[' + str(parameter[0]) + ',' + str(parameter[1]) + ',' + str(parameter[2]) + ',' + str(
            #     parameter[3]) + ']')
            # temp_best = all_temp_best_acc_list.max()
            # print("分數最高為 ", temp_best)

        class_names, train_points, train_labels = load_dataset(DATA_DIR)
        # x_train,x_test,y_train,y_test = Kfold_ramdom(train_points,train_labels,20,666,87)
        kf_times = 0
        temp_best_acc = 0
        temp_best_acc_list = []
        val_tf = False  # False True
        kf = KFold(n_splits=10, random_state=666, shuffle=True)

        if fit == 1 :
            if self.train_type =='nia_data':
                for train_index, test_index in kf.split(train_points):
                    x_val = []
                    y_val = []
                    kf_times += 1
                    np.random.seed(1688)
                    # print("KF",kf_times)
                    tf.compat.v1.reset_default_graph()
                    x_train, x_test = train_points[train_index], train_points[test_index]
                    y_train, y_test = train_labels[train_index], train_labels[test_index]
                    # print(len(x_train))
                    x_train = list(x_train)
                    y_train = list(y_train)

                    pop_index_list = np.random.choice(len(x_train), int(len(x_train) / 10), replace=False)
                    if val_tf == False:
                        # print("pass")
                        pass
                    elif val_tf == True:
                        for i in pop_index_list:
                            if i < len(x_train):
                                temp_list_pop = x_train.pop(i)
                                x_val.append(temp_list_pop)
                                temp_list_pop = y_train.pop(i)
                                y_val.append(temp_list_pop)
                    # print(len(x_val))
                    x_train = np.array(x_train)
                    y_train = np.array(y_train)
                    x_val = np.array(x_val)
                    y_val = np.array(y_val)
                    index = [i for i in range(len(x_train))]
                    np.random.seed(1688)
                    np.random.shuffle(index)
                    x_train = x_train[index]
                    y_train = y_train[index]
                    # 在KF第 1 fold 誤差最小為 : 0.1440217391304348 神經元為 [ 12.06658145 105.81916911 153.28292629 213.04487111]
                    if kf_times in kf_best_index:
                        parameter = all_best_fit[kf_best_index.index(kf_times)]  # [12.06658145 ,105.81916911, 153.28292629 ,213.04487111]
                        print(parameter)
                        kf_name = kf_times
                        use = False
                        if val_tf == False:  ##家這個出錯
                            if use == False:
                                print("是否用有val : ", val_tf)

                                history, preds, label = model_or(x_train, y_train, x_test, y_test, class_names, 32, parameter,kf_name)  # BATCH_SIZE,epochs
                                print (model_num)
                                model_num =model_num+1
                                print(model_num)

                                preds = np.argmax(preds, -1)  ## 分類
                                label_list.append(label)
                                preds_list.append(preds)
                                # history_list.append(history)
                                class_names_list.append(class_names)

                                same = 0
                                for i in range(len(preds)):
                                    if preds[i] == label[i]:
                                        same += 1
                                acc = same / len(preds)
                                print(acc, "正確率")
                                if acc > temp_best_acc:
                                    temp_best_acc = acc
                                temp_best_acc_list.append(temp_best_acc)

                                point = history.history['sparse_categorical_accuracy']
                                # print(point)
                                model_max_acc = max(point)
                                model_last_acc = point[-1]
                                print(model_max_acc, "or_model_max_acc")
                                print(model_last_acc, "or_model_last_acc")

                                history, preds, label = model(x_train, y_train, x_test, y_test, class_names, 32, parameter,kf_name)  # BATCH_SIZE,epochs
                                preds = np.argmax(preds, -1)  ## 分類
                                label_list.append(label)
                                preds_list.append(preds)
                                history_list.append(history)
                                class_names_list.append(class_names)

                                same = 0
                                for i in range(len(preds)):
                                    if preds[i] == label[i]:
                                        same += 1
                                acc = same / len(preds)
                                print(acc, "正確率")
                                if acc > temp_best_acc:
                                    temp_best_acc = acc
                                temp_best_acc_list.append(temp_best_acc)

                                point = history.history['sparse_categorical_accuracy']
                                # print(point)
                                model_max_acc = max(point)
                                model_last_acc = point[-1]
                                print(model_max_acc, "model_max_acc")
                                print(model_last_acc, "model_last_acc")
                                plt.rcParams['font.size'] = 14
                                # plot_confusion_matrix(label_list, preds_list, classes=class_names_list, normalize=False,title="confusion_matrix")
                                # plot_confusion_matrix(label_list, preds_list, classes=class_names_list, normalize=False)

                                # plot_history(history_list)
                                # plt.show()
                            elif use == True:
                                print(use)
                                preds, label = model_touse(x_train, y_train, x_test, y_test, class_names, 'pcm_predict_07170910',
                                                           parameter, i)  # BATCH_SIZE,epochs

                                preds = np.argmax(preds, -1)  ## 分類
                                # label_list.append(label)
                                preds_list.append(preds)
                                # history_list.append(history)
                                class_names_list.append(class_names)

                                same = 0
                                for i in range(len(preds)):
                                    if preds[i] == label[i]:
                                        same += 1
                                acc = same / len(preds)
                                print(acc, "正確率")
                                if acc > temp_best_acc:
                                    temp_best_acc = acc
                                temp_best_acc_list.append(temp_best_acc)

                                # point = history.history['sparse_categorical_accuracy']
                                # print(point)
                                # model_max_acc = max(point)
                                # model_last_acc = point[-1]
                                # print(model_max_acc,"model_max_acc")
                                # print(model_last_acc,"model_last_acc")
                                pass

                            # print(model_list[i])

                        elif val_tf == True:
                            pass
                            # print("是否用有val : ",val_tf)
                            # history,preds,label= model(x_train,y_train,x_val,y_val,class_names,32,parameter,i) #BATCH_SIZE,epochs
            elif self.train_type =='meal_data':
                x_val = []
                y_val = []
                kf_name = 'meal'
                for train_index, test_index in kf.split(train_points):
                    np.random.seed(1688)
                    tf.compat.v1.reset_default_graph()
                    x_train, x_test = train_points[train_index], train_points[test_index]
                    y_train, y_test = train_labels[train_index], train_labels[test_index]
                # x_train = train_points
                # y_train = train_labels

                print(len(x_train))

                x_train = list(x_train)
                y_train = list(y_train)

                np.random.seed(1688)
                pop_index_list = np.random.choice(len(x_train), int(len(x_train) / 10), replace=False)
                # for i in range (len(x_train)-int(len(x_train)/10)):
                #     if i%10 ==0:
                #         list_pop=x_train.pop(i)
                #         x_val.append(list_pop)
                #         list_pop=y_train.pop(i)
                #         y_val.append(list_pop)
                print(len(pop_index_list), "pop val ")
                for i in pop_index_list:
                    if i < len(x_train):
                        temp_list_pop = x_train.pop(i)
                        x_val.append(temp_list_pop)
                        temp_list_pop = y_train.pop(i)
                        y_val.append(temp_list_pop)
                print(len(x_val))
                x_train = np.array(x_train)
                y_train = np.array(y_train)
                x_val = np.array(x_val)
                y_val = np.array(y_val)

                index = [i for i in range(len(x_train))]
                np.random.seed(1688)
                np.random.shuffle(index)
                x_train = x_train[index]
                y_train = y_train[index]
                if self.train_or :
                    history, preds, label = model_or(x_train, y_train, x_val, y_val, class_names, 32, parameter,kf_name,self.save_name)  # BATCH_SIZE,epochs
                

                    preds = np.argmax(preds, -1)  ## 分類
                    label_list.append(label)
                    preds_list.append(preds)
                    history_list.append(history)
                    class_names_list.append(class_names)

                    same = 0
                    for i in range(len(preds)):
                        if preds[i] == label[i]:
                            same += 1
                    or_acc = same / len(preds)
                    print(or_acc, "正確率")
                    if or_acc > temp_best_acc:
                        temp_best_acc = or_acc
                    temp_best_acc_list.append(temp_best_acc)

                    point = history.history['sparse_categorical_accuracy']
                    # print(point)
                    model_max_acc = max(point)
                    model_last_acc = point[-1]
                    print(model_max_acc, "or_model_max_acc")
                    print(model_last_acc, "or_model_last_acc")
                    print(or_acc)
                    model_num = model_num + 1
                    print(model_num)
                if model_num >1 :
                    # if  self.train_or :
                    for i in range (model_num):
                        print(i)
                        parameter = best_fit[i]
                        kf_name = model_name[i]
                        history, preds, label = model(x_train, y_train, x_val, y_val, class_names, 32, parameter,kf_name,self.save_name)  # BATCH_SIZE,epochs
                        preds = np.argmax(preds, -1)  ## 分類
                        label_list.append(label)
                        preds_list.append(preds)
                        history_list.append(history)
                        class_names_list.append(class_names)

                        same = 0
                        for i in range(len(preds)):
                            if preds[i] == label[i]:
                                same += 1
                        acc = same / len(preds)
                        print(acc, "正確率")
                        if acc > temp_best_acc:
                            temp_best_acc = acc
                        temp_best_acc_list.append(temp_best_acc)

                        point = history.history['sparse_categorical_accuracy']
                        # print(point)
                        model_max_acc = max(point)
                        model_last_acc = point[-1]
                        print(model_max_acc, "model_max_acc")
                        print(model_last_acc, "model_last_acc")
                elif model_num ==1:
                    history, preds, label = model(x_train, y_train, x_val, y_val, class_names, 32, parameter,kf_name,self.save_name)  # BATCH_SIZE,epochs
                    preds = np.argmax(preds, -1)  ## 分類
                    label_list.append(label)
                    preds_list.append(preds)
                    history_list.append(history)
                    class_names_list.append(class_names)

                    same = 0
                    for i in range(len(preds)):
                        if preds[i] == label[i]:
                            same += 1
                    acc = same / len(preds)
                    print(acc, "正確率")
                    if acc > temp_best_acc:
                        temp_best_acc = acc
                    temp_best_acc_list.append(temp_best_acc)

                    point = history.history['sparse_categorical_accuracy']
                    # print(point)
                    model_max_acc = max(point)
                    model_last_acc = point[-1]
                    print(model_max_acc, "model_max_acc")
                    print(model_last_acc, "model_last_acc")
                # plt.rcParams['font.size'] = 20
                fig, ax = plt.subplots()
                plt.rcParams['font.size'] = 20

                for i in range (len(all_temp_best_acc_list)):
                    ax.plot(all_temp_best_acc_list[i], color=color[i])
                    # print(model_name)
                    list(model_name).insert(0,['or'])
                    # print(model_name)
                    ax.legend(model_name, loc='lower right')
                    print(max(all_temp_best_acc_list[i]))
                    ax.set(title='learning curve')
            plt.rcParams['font.size'] = 16
            plot_history(history_list,model_num)
            plt.rcParams['font.size'] = 20

            plot_confusion_matrix(label_list, preds_list,model_num, classes=class_names_list, normalize=False)


        
        elif fit == 0 :
            print('pass')

        for i in range(model_num):
            algo_best = np.max(all_buf_acc_list[i])#
            all_buf_acc_list_index = np.where(all_buf_acc_list[i] == algo_best)
            # print(all_buf_fit_list[i][all_buf_acc_list_index][0])
            best_fit.append(all_buf_fit_list[i][all_buf_acc_list_index][0])
            parameter = best_fit[i]  ####下面要改 可以2跑兩個演算法的    best_fit[0]#all_best_fit
            print()
            print('model_name : ', model_name[i])

            print("parameter : ")

            print('[' + str(parameter[0]) + ',' + str(parameter[1]) + ',' + str(parameter[2]) + ',' + str(
                parameter[3]) + ']')
            print("分數最高為 ", algo_best)
            print("train 分數最高為 ", temp_best_acc_list[i])

        if self.show :
            plt.show()


