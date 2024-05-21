# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import os
import glob
import time
import datetime
# import trimesh
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
import os

vram=1024*10#1024*3
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=vram)])
  except RuntimeError as e:
    print(e)
import sys

sys.path.append('../')
# End of fix
# DATA_DIR ="C:/Users/ccu/OneDrive/桌面/one_pycharm/npy"
DATA_DIR ="C:/Users/karta/OneDrive/桌面/one_pycharm/npy"

color = [[0.5,0,0],'b','g','r','y','k','c','m','gray','lightgray']
from niapy.algorithms.basic import ParticleSwarmAlgorithm
from niapy.task import Task

from mealpy import FloatVar, ARO,PSO,TPO,SA,BA

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
def parse_dataset(num_points,DATA_DIR):
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_names = []

    train_name = os.path.basename(DATA_DIR)
    print("train_name :",train_name)
    #return train_name
    save_name_p= train_name+"_train_points.npy"
    save_name_l= train_name+"_train_labels.npy"
    folders = glob.glob(os.path.join(DATA_DIR, "*"))
    print(folders)
    for i, folder in enumerate(folders):
        temp = os.path.basename(folder)
        print("processing class: {}".format(temp))
        class_names.append(temp)
        train_files = glob.glob(os.path.join(folder, "*"))
        if train_name =="npy" or train_name =="npp" :
            for f in train_files:
                train_points.append(trimesh.load(trimesh.PointCloud(np.load(f)).convex_hull).sample(num_points))
                #print(len(train_points[i])) #2048
                train_labels.append(i)
        else:
            for f in train_files:
                train_points.append(trimesh.load(f).sample(num_points))
                train_labels.append(i)
    np.save(save_name_p, train_points)
    np.save(save_name_l, train_labels)
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
    save_name_p ="5_800_all_obj.npy" #"all_obj.npy"
    save_name_l ="5_800_all_label.npy" #"all_label.npy"
    # base_path = "C:/Users/ccu/OneDrive/桌面/one_pycharm"
    base_path = "C:/Users/karta/OneDrive/桌面/one_pycharm"


    if not os.path.exists(base_path+'/'+save_name_p):
        print("not exists")
        # class_names,train_points,train_labels = parse_dataset(2048,DATA_DIR)

    else: #  抓種類資料的也要改  如果都用固定npy 沒有給他自己產生 不要用到data_dir
        print("exists")
        folders = glob.glob(os.path.join(DATA_DIR, "*"))
        for i, folder in enumerate(folders):
            temp = os.path.basename(folder)
            print("processing class: {}".format(temp))
            class_names.append(temp)
            train_points = np.load(save_name_p, allow_pickle=True)
            train_labels = np.load(save_name_l, allow_pickle=True)
        class_names = np.array(class_names)
    return class_names,train_points,train_labels

ttime = datetime.datetime.now()
ttime = str(ttime)
print(ttime)

Neurons_list=[16,32,64,128,256]
Dropout_list=[0,0.1,0.2,0.3,0.4]
learnnig_curve = []
all_learnnig_curve = []
def model(x_train,y_train,x_test,y_test,class_names,batch_size,parameter):
    print("fittt")
    Dropout_0 = 0.3
    Neurons_1 = int(parameter[0]) #Neurons_list[int(parameter[1])]
    Neurons_2 = 32
    Neurons_3 = Neurons_2*2
    Neurons_4 = int(parameter[1])
    Neurons_5 = int(parameter[2])
    Neurons_6 = int(parameter[3])

    #print(Dropout_0,Neurons_1,Neurons_2,Neurons_3,Neurons_4,Neurons_5,Neurons_6,Neurons_7)
    print(" tnet :","3",'\n',"conv_bn :",Neurons_1,'\n',"conv_bn :",Neurons_2,'\n',"tnet :",Neurons_2,'\n',"conv_bn :",Neurons_2,'\n'
    ,"conv_bn :",Neurons_3,'\n',"conv_bn :",Neurons_4,'\n',"GlobalMaxPooling1D",'\n',"dense_bn :",Neurons_5,'\n'
    ,"Dropout :",Dropout_0,'\n',"dense_bn :",Neurons_6,'\n',"Dropout :",Dropout_0,'\n',"output_Dense :",len(class_names),'\n')

    inputs = tf.keras.Input(shape=(20, 3))
    ##  orginal model
    x = tnet(inputs, 3)
    x = conv_bn(x, Neurons_1)
    x = conv_bn(x, Neurons_2) #32
    x = yomatnet(x, Neurons_2) #32
    x = conv_bn(x, Neurons_2) #32
    x = conv_bn(x, Neurons_3) # 64
    x = conv_bn(x, Neurons_4)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, Neurons_5)
    x = layers.Dropout(Dropout_0)(x) # 0.3
    x = dense_bn(x, Neurons_6)
    x = layers.Dropout(Dropout_0)(x) # 0.3

    outputs = layers.Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
    # model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["sparse_categorical_accuracy"], )
    history = model.fit(x_train, y_train,batch_size=batch_size, epochs=10, verbose=0)
    preds = model.predict(x_test)
    label = y_test
    return history,preds,label
#"bounds": FloatVar(lb=[0,0.1,0.2,0.3,0.4], ub=[0,0.1,0.2,0.3,0.4], name="delta")
def objective_function(solution):
    history,preds,label= model(x_train,y_train,x_val,y_val,class_names,32,solution) #BATCH_SIZE,epochs
    preds = np.argmax(preds, -1) ## 分類
    same = 0
    global temp_best_acc
    for i in range (len(preds)):
        if preds[i] == label[i]:
            same+=1
    acc = same/len(preds)
    print(acc,"正確率")
    if acc > temp_best_acc :
        temp_best_acc = acc
    temp_best_acc_list.append(temp_best_acc)
    buf_acc_list.append(acc)
    buf_fit_list.append(solution)
    point = history.history['sparse_categorical_accuracy']
    model_max_acc = max(point)
    model_last_acc = point[-1]
    print(model_max_acc,"model_max_acc")
    print(model_last_acc,"model_last_acc")
    # error = 1 - max(point)
    # print(error,'error')
    error = 1 - acc
    #print(error,"ERROR")
    print()
    return error

problem_dict = {
    "bounds": FloatVar(lb=[1, 1, 1, 1], ub=[257, 257, 257, 257], name="delta"),

    "obj_func": objective_function,
    "minmax": "min",
}

# g_best = model.solve(problem_dict)





# algo = ParticleSwarmAlgorithm(population_size=100, min_velocity=-4.0, max_velocity=4.0)


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

class_names,train_points,train_labels = load_dataset(DATA_DIR)
x_val = []
y_val = []
print(class_names,len(train_points),len(train_labels))
# x_train,x_test,y_train,y_test = Kfold_ramdom(train_points,train_labels,20,666,87)
kf_times = 0
all_best_fit = []
all_best = []
all_temp_best_acc_list = []
all_buf_acc_list = []
all_buf_fit_list = []
kf = KFold(n_splits=10, random_state=666, shuffle=True)

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

#np.random.seed(1688)
pop_index_list = np.random.choice(len(x_train),int(len(x_train)/10),replace=False)
# for i in range (len(x_train)-int(len(x_train)/10)):
#     if i%10 ==0:
#         list_pop=x_train.pop(i)
#         x_val.append(list_pop)
#         list_pop=y_train.pop(i)
#         y_val.append(list_pop)
print(len(pop_index_list),"pop val ")
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
#print(len(x_train),len(x_val),len(x_test))
# index = [i for i in range(len(x_train))]
# np.random.shuffle(index)
# x_train = x_train[index]
# y_train = y_train[index]
model_name = []

# task = Task(problem=Griewank(dimension=4, lower=1, upper=257), max_evals=20)
# best_fit,best = algo.run(task=task) # 每個fold 的最好

def model0(problem_dict,epoch):
    print('PSO.AIW_PSO')
    model_name.append('PSO')
    model = PSO.AIW_PSO(epoch=epoch, pop_size=5, c1=2.05, c2=2.05, alpha=0.4)
    g_best = model.solve(problem_dict,termination=term_dict1)
    return g_best
def model1(problem_dict,epoch):
    print('ARO.IARO')
    model_name.append('ARO')

    model = ARO.IARO(epoch=epoch, pop_size=5)
    g_best = model.solve(problem_dict,termination=term_dict1)
    return g_best
# mmodel = ARO.IARO(epoch=20, pop_size=5)
def model2(problem_dict,epoch):
    print('TPO.DevTPO')
    model_name.append('TPO')
    model = TPO.DevTPO(epoch=int(epoch), pop_size=5, alpha=0.3, beta=50., theta=0.9)
    g_best = model.solve(problem_dict,termination=term_dict1)
    return g_best
def model3(problem_dict,epoch):
    print('SA.GaussianSA')
    model_name.append('SA')
    model = SA.GaussianSA(epoch=epoch, pop_size=5, temp_init = 100, cooling_rate = 0.99, scale = 0.1)
    g_best = model.solve(problem_dict,termination=term_dict1)
    return g_best
def model4(problem_dict,epoch):
    print('BA.AdaptiveBA')
    model_name.append('BA')
    model = BA.AdaptiveBA(epoch=epoch, pop_size=5, loudness_min = 1.0, loudness_max = 2.0)#, pr_min = -2.5, pr_max = 0.85, pf_min = 0.1, pf_max = 10.)
    g_best = model.solve(problem_dict,termination=term_dict1)
    return g_best
model_type= ['pso','aro','tpo','sa','ba']
if len(model_type)>1:
    for i in range (len(model_type)):
        # model = PSO.AIW_PSO(epoch=20, pop_size=5, c1=2.05, c2=2.05, alpha=0.4)
        # model = ARO.IARO(epoch=20, pop_size=5)
        temp_best_acc = 0
        temp_best_acc_list = []
        buf_acc_list = []
        buf_fit_list = []
        meal_model = eval("model"+str(i))
        epoch = 100
        term_dict1 = {
            "max_epoch": epoch,
            "max_fe": epoch,  # 100000 number of function evaluation
            "max_time": int(epoch*200),  # 10 seconds to run the program
            "max_early_stop": epoch  # 15 epochs if the best objective_function is not getting better we stop the program
        }
        g_best = meal_model(problem_dict,30)

        print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
        print(len(np.array(temp_best_acc_list)),'temp_best_acc_list')
        print(len(np.array(buf_acc_list)),'buf_acc_list')
        print(len(np.array(buf_fit_list)),'buf_fit_list')

        all_temp_best_acc_list.append(np.array(temp_best_acc_list)[0:epoch])
        all_buf_acc_list.append(np.array(buf_acc_list)[0:epoch])
        all_buf_fit_list.append(np.array(buf_fit_list)[0:epoch])
    fig, ax = plt.subplots()

    for i in range(len(all_temp_best_acc_list)):
        ax.plot(all_temp_best_acc_list[i], color=color[i])
        ax.legend(model_name, loc='lower right')

        ax.set(title='lc_'+" meal _"+str(ttime[5:7]+ttime[8:10]+ttime[11:13]+ttime[14:16])+'_acc_')
        # 是否保存pso aro 演算法list
elif len(model_type)==1:

    meal_model = PSO.AIW_PSO(epoch=20, pop_size=5, c1=2.05, c2=2.05, alpha=0.4)


    g_best = meal_model.solve(problem_dict)
    print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    print(f"Solution: {meal_model.g_best.solution}, Fitness: {meal_model.g_best.target.fitness}")
    #print(best_fit,"best_fitttttt")
    #print(best,"besttttt")
    # print(g_best.solution.type)
    # all_best_fit.append(list(g_best.solution))
    # all_best.append(list(g_best.target.fitness))
    print(len(temp_best_acc_list),'temp_best_acc_list')
    all_temp_best_acc_list.append(temp_best_acc_list)
    all_buf_acc_list.append(buf_acc_list)

    all_buf_fit_list.append(buf_fit_list)
    # all_best_fit.append(g_best.solution)
    #print(all_temp_best_acc_list)

    # print(all_best_fit)
    # print(all_best)
    fig, ax = plt.subplots()
    print(ttime)
    for i in range (len(all_temp_best_acc_list)):
        ax.plot(all_temp_best_acc_list[i], color=color[i])
        ax.set(title='lc_'+" meal _"+str(ttime[5:7]+ttime[8:10]+ttime[11:13]+ttime[14:16])+'_acc_') # str(np.array(all_temp_best_acc_list).max())
save_name = "meal_"+ttime[5:7]+ttime[8:10]+ttime[11:13]+ttime[14:16]
np.save('./meal_data/'+str(save_name)+'model_name', model_name)

np.save('./meal_data/'+str(save_name)+'acc', all_temp_best_acc_list)
np.save('./meal_data/'+str(save_name)+'bufacc', all_buf_acc_list)
np.save('./meal_data/'+str(save_name)+'fit', all_buf_fit_list)
# for i in range (len(all_best)) :
#     if all_best[i]  == min(all_best):
#         print("在KF第",i+1,"fold","誤差最小為 :",all_best[i],"神經元為",all_best_fit[i])
#         ax.set(title='learning curve'+str(i+1)+"_"+str(all_best_fit[i]))
# ax.legend(['1','2','3','4','5','6','7','8','9','10'], loc='upper left')
plt.savefig('./meal_data/'+str(save_name)+'.png')
plt.show()
# kf = KFold(n_splits=10, random_state=666, shuffle=True)
# for train_index, test_index  in kf.split(train_points):
#     x_val = []
#     y_val = []
#     kf_times +=1
#     print("KF",kf_times)
#     tf.compat.v1.reset_default_graph()
#     x_train, x_test = train_points[train_index], train_points[test_index]
#     y_train, y_test = train_labels[train_index], train_labels[test_index]
#     print(len(x_train))
#     x_train = list(x_train)
#     y_train = list(y_train)
#     #np.random.seed(1688)
#     pop_index_list = np.random.choice(len(x_train),int(len(x_train)/10),replace=False)
#     # for i in range (len(x_train)-int(len(x_train)/10)):
#     #     if i%10 ==0:
#     #         list_pop=x_train.pop(i)
#     #         x_val.append(list_pop)
#     #         list_pop=y_train.pop(i)
#     #         y_val.append(list_pop)
#     print(len(pop_index_list))
#     for i in pop_index_list:
#         if i < len(x_train):
#             temp_list_pop = x_train.pop(i)
#             x_val.append(temp_list_pop)
#             temp_list_pop = y_train.pop(i)
#             y_val.append(temp_list_pop)
#     print(len(x_val))
#     x_train = np.array(x_train)
#     y_train = np.array(y_train)
#     x_val = np.array(x_val)
#     y_val = np.array(y_val)
#     #print(len(x_train),len(x_val),len(x_test))
#     index = [i for i in range(len(x_train))]
#     np.random.shuffle(index)
#     x_train = x_train[index]
#     y_train = y_train[index]
#     temp_best_acc_list= []

#     # task = Task(problem=Griewank(dimension=4, lower=1, upper=257), max_evals=20)
#     # best_fit,best = algo.run(task=task) # 每個fold 的最好

#     mmodel = ARO.IARO(epoch=200, pop_size=50)
#     temp_best_acc = 0
#     g_best = mmodel.solve(problem_dict)
#     print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
#     print(f"Solution: {mmodel.g_best.solution}, Fitness: {mmodel.g_best.target.fitness}")
#     #print(best_fit,"best_fitttttt")
#     #print(best,"besttttt")
#     all_best_fit.append(best_fit)
#     all_best.append(best)
#     all_temp_best_acc_list.append(temp_best_acc_list)
#     #print(all_temp_best_acc_list)

# # print(all_best_fit)
# # print(all_best)
# fig, ax = plt.subplots()
# print(ttime)

# ax.set(title='learning curve')
# save_name = "kf_nia_"+ttime[5:7]+ttime[8:10]+ttime[11:13]+ttime[14:16]
# np.save(str(save_name)+'acc', all_temp_best_acc_list)
# np.save(str(save_name)+'fit', all_best_fit)
# for i in range (len(all_temp_best_acc_list)):
#     ax.plot(all_temp_best_acc_list[i], color=color[i])
# for i in range (len(all_best)) :
#     if all_best[i]  == min(all_best):
#         print("在KF第",i+1,"fold","誤差最小為 :",all_best[i],"神經元為",all_best_fit[i])
#         ax.set(title='learning curve'+str(i+1)+"_"+str(all_best_fit[i]))
# ax.legend(['1','2','3','4','5','6','7','8','9','10'], loc='upper left')
# plt.savefig(str(save_name)+'.png')