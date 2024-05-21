import time

import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression
from skopt import gp_minimize
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
import os

def target_change(source,zoom_x=1,zoom_y=1,translate_x=0,translate_y=0,rotate=0):
    target_buffer = source.copy()

    target_buffer[:,0]*=zoom_x
    target_buffer[:,1]*=zoom_y

    target_buffer[:,0]+=translate_x
    target_buffer[:,1]+=translate_y

    theta = np.radians(rotate)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    target_buffer = np.dot(target_buffer, rotation_matrix)
    return target_buffer

def target_change_back(source,zoom_x=1,zoom_y=1,translate_x=0,translate_y=0,rotate=0):
    target_buffer = source.copy()

    theta = np.radians(rotate)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    target_buffer = np.dot(target_buffer, rotation_matrix)

    target_buffer[:,0]+=translate_x
    target_buffer[:,1]+=translate_y

    target_buffer[:, 0] *= zoom_x
    target_buffer[:, 1] *= zoom_y


    return target_buffer

def show_plt(x,y,tittle="?"):
    plt.scatter(x, y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(tittle)
def Fit_show(source_np,target_np,show=False):
    def LR_degree(points):

        x = points[:, 0].reshape(-1, 1)
        y = points[:, 1]

        model = LinearRegression().fit(x, y)

        # 計算回歸線的斜率和截距
        slope = model.coef_[0]
        intercept = model.intercept_
        # 計算夾角（弧度）
        angle_radians = np.arctan(slope)

        # 將弧度轉換為度
        angle_degrees = np.degrees(angle_radians)

        # print(angle_degrees, intercept)

        # 繪製散點圖和回歸線
        # if show:
            # plt.scatter(points[:, 0], points[:, 1], label='Data points')
            # plt.plot(points[:, 0], model.predict(points[:, 0].reshape(-1, 1)), color='red', label='Regression line')
            # plt.xlabel('X')
            # plt.ylabel('Y')
            # plt.title('Linear Regression')
            # plt.legend()
            # plt.show()

        return angle_degrees
    def get_distance(source, target):
        kdtree_Y = KDTree(target)
        distances, indices = kdtree_Y.query(source)
        mean_distance = np.mean(distances)
        return mean_distance
    def calculate_angle(point, centroid):
        # Calculate the vector from centroid to the point
        vector = point - centroid

        # Calculate the angle with the x-axis using arctangent
        angle = np.arctan2(vector[1], vector[0])

        return np.degrees(angle)
    def rotate_sort_algorthm(input_array, closest,try_close=False):

        input_array_centroid = np.mean(input_array, axis=0)
        # print(input_array_centroid)
        #抓兩個的中心點 是沒錯的
        #角度看起來是沒問題的 主要是它會抓距離近的?
        angles = np.array([calculate_angle(point, input_array_centroid) for point in input_array])
        angles = -(np.array([calculate_angle(point, input_array_centroid) for point in input_array])- angles[closest])
        angles[angles < 0] += 360
        print(angles)
        close_angle=[]
        if try_close :
            for i in range(len(angles)):
                for j in range(len(angles)):
                    if abs(angles[i]-angles[j] )<1 and i!=j :
                        close_angle.append([i,j])
            print(close_angle)
            sorted_indices =list(np.argsort(angles))
            out_indices=[]
            print(sorted_indices)
            if close_angle !=[]:
                print('有相近 可能有兩個sorted_indices 要試試看')
                for i in range(len(close_angle)):#int(len(close_angle)/2)

                    change_1 ,change_2 = close_angle[i]
                    change_1_index = sorted_indices.index(change_1)
                    change_2_index = sorted_indices.index(change_2)


                    sorted_indices[change_1_index] = change_2
                    sorted_indices[change_2_index] = change_1
                    bufsorted_indices = sorted_indices.copy()
                    print(bufsorted_indices)

                    out_indices.append(bufsorted_indices)
            print(out_indices)
        # sorted_xs = input_array[sorted_indices]
        else:
            out_indices = np.argsort(angles)

        if show:
            for i, angle in enumerate(angles):
                plt.text(input_array[i, 0], input_array[i, 1], f'{  i}', fontsize=8, ha='left')
        # plt.show()
        return out_indices
    def fit_to_target(source, target ,closest_source,closest_target):
        test_dis = 0
        source_indexs = rotate_sort_algorthm(source, closest_source,try_close=1)
        target_index = rotate_sort_algorthm(target, closest_target,try_close=0)
        # print(source_index)
        # print(target_index)
        # 這邊把 indedx連起來 應該是 上面的
        source_fit = source.copy()
        print(source_indexs)
        if len(source_indexs)==1:
            for i in range(np.shape(source_fit)[0]):
                source_fit[source_indexs[i]] = target[target_index[i]]
            for i in range(len(source_fit)):
                x1, y1 = source_fit[i, 0], source_fit[i, 1]
                x2, y2 = source[i, 0], source[i, 1]
                distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                test_dis = test_dis + distance
                if show:
                    plt.plot([source_fit[i, 0], source[i, 0]], [source_fit[i, 1], source[i, 1]], color='black')
            print(test_dis)

            if show:
                plt.show()

            return source_fit,test_dis
        elif len(source_indexs) > 1:
            fit_dis = []
            buf_fit_source = []
            for i in range(len(source_indexs)):
                source_index = source_indexs[i]
                test_dis = 0
                print(source_index)
                for i in range(np.shape(source_fit)[0]):
                    source_fit[source_index[i]] = target[target_index[i]]
                buf_fit_source.append(source_fit.copy())
                for i in range(len(source_fit)):
                    x1, y1 = source_fit[i, 0], source_fit[i, 1]
                    x2, y2 = source[i, 0], source[i, 1]
                    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    test_dis = test_dis + distance

                #     plt.plot([source_fit[i, 0], source[i, 0]], [source_fit[i, 1], source[i, 1]], color='black')
                # plt.show()
                fit_dis.append(test_dis)
                print(fit_dis)
            tt = fit_dis.index(min(fit_dis))
            print(tt)
            source_fi=buf_fit_source[tt].copy()

            if show:
                for i in range(len(source_fi)):
                    plt.plot([source_fi[i, 0], source[i, 0]], [source_fi[i, 1], source[i, 1]], color='black')
                plt.show()


            return buf_fit_source[tt], fit_dis[tt]
    print('start')

    

    if len(source_np)!=len(target_np) and os.path.exists("last_point.npy") == 1:
        source_last_fit=np.load("last_point.npy")
        # target_centroid = np.mean(target_np, axis=0)
        # source_last_fit_centroid = np.mean(source_last_fit, axis=0)
        # distance=target_centroid-source_last_fit_centroid
        # source_last_fit_x = source_last_fit[:, 0]-distance[0]
        # source_last_fit_y = source_last_fit[:, 1]-distance[1]

        # source_fit = np.vstack((source_last_fit_x, source_last_fit_y)).T
        # return source_fit
        return source_last_fit
    else:
        source_x = source_np[:, 0]
        source_y = source_np[:, 1]
        target_x = target_np[:, 0]
        target_y = target_np[:, 1]

        if show:
            #原始位置
            show_plt(target_x, target_y)
            show_plt(source_x, source_y)
            plt.show()

        #平移至中心點################################################################################################################################################
        source_centroid = np.mean(source_np, axis=0)
        target_centroid = np.mean(target_np, axis=0)
        # print('source_centroid',source_centroid)
        # print('target_centroid',target_centroid)

        source_centering_x=source_x-source_centroid[0]
        source_centering_y=source_y-source_centroid[1]

        target_centering_x=target_x-target_centroid[0]
        target_centering_y=target_y-target_centroid[1]

        panning_x=target_centroid[0]
        panning_y=target_centroid[1]
        print('平移中心點')

        if show:
            show_plt(target_centering_x, target_centering_y)
            show_plt(source_centering_x, source_centering_y)
            plt.show()

        #第一次旋轉################################################################################################################################################
        source_centering = np.vstack((source_centering_x, source_centering_y)).T
        target_centering = np.vstack((target_centering_x,target_centering_y)).T

        source_angle_degrees = LR_degree(source_centering)
        target_angle_degrees = LR_degree(target_centering)

        rotate_1=source_angle_degrees-target_angle_degrees
        print('第一次旋轉 : ',rotate_1,'度')
        source_panning_rotation_final = target_change(source_centering, zoom_x=1, zoom_y=1, translate_x=0,
                                                                           translate_y=0, rotate=rotate_1)


        source_final_x = source_panning_rotation_final[:, 0]
        source_final_y = source_panning_rotation_final[:, 1]

        target_centering_x = target_centering[:, 0]
        target_centering_y = target_centering[:, 1]

        if show:
            target_centroid = np.mean(target_centering, axis=0)
            source_centroid = np.mean(source_panning_rotation_final, axis=0)


            plt.scatter(source_centroid[0], source_centroid[1], color='red', marker='x', label='Centroid')
            plt.scatter(target_centroid[0], target_centroid[1], color='blue', marker='x', label='Centroid')
            show_plt(target_centering_x, target_centering_y)
            show_plt(source_final_x, source_final_y)
            plt.show()

        source_final = np.vstack((source_final_x,source_final_y)).T
        
        source_gaussian=source_final

        source_gaussian_x = source_gaussian[:, 0]
        source_gaussian_y = source_gaussian[:, 1]
        source_gaussian_x=source_gaussian_x+panning_x
        source_gaussian_y=source_gaussian_y+panning_y
        source_gaussian = np.vstack((source_gaussian_x, source_gaussian_y)).T
        source_gaussian_x = source_gaussian[:, 0]
        source_gaussian_y = source_gaussian[:, 1]
        if show:
            show_plt(target_x, target_y)
            show_plt(source_gaussian_x, source_gaussian_y)
            # plt.show()
        distances = cdist(source_gaussian, target_np)
        # print(distances)
        min_index = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
        # print(min_index[0],min_index[1])
        if show:
            show_plt(target_np[min_index[1]][0],target_np[min_index[1]][1])
            show_plt(source_gaussian[min_index[0]][0], source_gaussian[min_index[0]][1])
            # plt.show()
            print('find min')

        # print(target_np)
        source_fit,test_dis=fit_to_target(source_gaussian,target_np,closest_source=min_index[0],closest_target=min_index[1])#
        np.save("last_point.npy",source_fit)

    return source_fit,test_dis

def Fit(source_np,target_np,show=False,show_f=False):
    def LR_degree(points):

        x = points[:, 0].reshape(-1, 1)
        y = points[:, 1]

        model = LinearRegression().fit(x, y)

        # 計算回歸線的斜率和截距
        slope = model.coef_[0]
        intercept = model.intercept_
        # 計算夾角（弧度）
        angle_radians = np.arctan(slope)

        # 將弧度轉換為度
        angle_degrees = np.degrees(angle_radians)

        # print(angle_degrees, intercept)

        # 繪製散點圖和回歸線
        # if show:
        #     plt.scatter(points[:, 0], points[:, 1], label='Data points')
        #     plt.plot(points[:, 0], model.predict(points[:, 0].reshape(-1, 1)), color='red', label='Regression line')
        #     plt.xlabel('X')
        #     plt.ylabel('Y')
        #     plt.title('Linear Regression')
        #     plt.legend()
        #     plt.show()

        return angle_degrees
    def get_distance(source, target):
        kdtree_Y = KDTree(target)
        distances, indices = kdtree_Y.query(source)
        mean_distance = np.mean(distances)
        return mean_distance
    def calculate_angle(point, centroid):
        # Calculate the vector from centroid to the point
        vector = point - centroid

        # Calculate the angle with the x-axis using arctangent
        angle = np.arctan2(vector[1], vector[0])

        return np.degrees(angle)
    def rotate_sort_algorthm(input_array, closest,try_close=False):

        input_array_centroid = np.mean(input_array, axis=0)
        # print(input_array_centroid)
        #抓兩個的中心點 是沒錯的
        #角度看起來是沒問題的 主要是它會抓距離近的?
        angles = np.array([calculate_angle(point, input_array_centroid) for point in input_array])
        angles = -(np.array([calculate_angle(point, input_array_centroid) for point in input_array])- angles[closest])
        angles[angles < 0] += 360
        # print(angles)
        close_angle=[]
        if try_close :
            for i in range(len(angles)):
                for j in range(len(angles)):
                    if abs(angles[i]-angles[j] )<1 and i!=j :
                        close_angle.append([i,j])
            # print(close_angle)
            sorted_indices =list(np.argsort(angles))
            out_indices=[]
            # print(sorted_indices)
            if close_angle !=[]:
                # print('有相近 可能有兩個sorted_indices 要試試看')
                for i in range(len(close_angle)):#int(len(close_angle)/2)

                    change_1 ,change_2 = close_angle[i]
                    change_1_index = sorted_indices.index(change_1)
                    change_2_index = sorted_indices.index(change_2)


                    sorted_indices[change_1_index] = change_2
                    sorted_indices[change_2_index] = change_1
                    bufsorted_indices = sorted_indices.copy()
                    # print(bufsorted_indices)

                    out_indices.append(bufsorted_indices)
            else:
                out_indices = [list(np.argsort(angles))]
            # print(out_indices)
        # sorted_xs = input_array[sorted_indices]
        else:
            out_indices = np.argsort(angles)

        if show:
            for i, angle in enumerate(angles):
                plt.text(input_array[i, 0], input_array[i, 1], f'{  i}', fontsize=8, ha='left')
        # plt.show()
        return out_indices
    def fit_to_target(source, target ,closest_source,closest_target):
        test_dis = 0
        source_indexs = rotate_sort_algorthm(source, closest_source,try_close=1)
        target_index = rotate_sort_algorthm(target, closest_target,try_close=0)
        # print(source_index)
        # print(target_index)
        # 這邊把 indedx連起來 應該是 上面的
        source_fit = source.copy()
        # print(source_indexs)
        # print(len(target_index))
        if len(source_indexs)==1:
            source_index = source_indexs[0]
            for i in range(np.shape(source_fit)[0]):
                source_fit[source_index[i]] = target[target_index[i]]
            for i in range(len(source_fit)):
                x1, y1 = source_fit[i, 0], source_fit[i, 1]
                x2, y2 = source[i, 0], source[i, 1]
                distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                test_dis = test_dis + distance
                if show:
                    plt.plot([source_fit[i, 0], source[i, 0]], [source_fit[i, 1], source[i, 1]], color='black')
            # print(test_dis)

            if show:
                plt.show()

            return source_fit,test_dis,source_index,target_index
        elif len(source_indexs) > 1:
            fit_dis = []
            buf_fit_source = []
            for i in range(len(source_indexs)):
                source_index = source_indexs[i]
                test_dis = 0
                for i in range(np.shape(source_fit)[0]):
                    source_fit[source_index[i]] = target[target_index[i]]
                    temp=source_fit.copy()
                buf_fit_source.append(temp)

                for i in range(len(source_fit)):
                    x1, y1 = source_fit[i, 0], source_fit[i, 1]
                    x2, y2 = source[i, 0], source[i, 1]
                    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    test_dis = test_dis + distance
                    if show:
                        plt.plot([source_fit[i, 0], source[i, 0]], [source_fit[i, 1], source[i, 1]], color='black')
                fit_dis.append(test_dis)
            tt = fit_dis.index(min(fit_dis))
            # 順序最小 source_indexs[tt]
            if show:
                plt.show()
            # print(buf_fit_source[tt], fit_dis[tt])
            return buf_fit_source[tt], fit_dis[tt],source_indexs[tt],target_index
    print('start')


    try_angle = list(range(0,360,int(360/len(source_np))))
    # print(try_angle)
    fit_dis = []
    source_inds=[]
    target_inds=[]
    FIT_piont_list=[]
    source_npb=source_np
    for i in try_angle:#[10,20,30,40,50,60,70,80,90,100,110,120,130,140]:
        # print(i,'旋轉測試')
        source_np = target_change(source_npb, zoom_x=1.0, zoom_y=1.0, translate_x=0,translate_y=0, rotate=i)

        if len(target_np)!=len(source_np) and os.path.exists("last_point.npy") == 1:
            source_last_fit=np.load("last_point.npy")
            # target_centroid = np.mean(target_np, axis=0)
            # source_last_fit_centroid = np.mean(source_last_fit, axis=0)
            # distance=target_centroid-source_last_fit_centroid
            # source_last_fit_x = source_last_fit[:, 0]-distance[0]
            # source_last_fit_y = source_last_fit[:, 1]-distance[1]

            # source_fit = np.vstack((source_last_fit_x, source_last_fit_y)).T
            # return source_fit
            return source_last_fit,0
        else:
            source_x = source_np[:, 0]
            source_y = source_np[:, 1]
            target_x = target_np[:, 0]
            target_y = target_np[:, 1]

            if show:
                #原始位置
                show_plt(target_x, target_y)
                show_plt(source_x, source_y)
                plt.show()

            #平移至中心點################################################################################################################################################
            source_centroid = np.mean(source_np, axis=0)
            target_centroid = np.mean(target_np, axis=0)
            # print('source_centroid',source_centroid)
            # print('target_centroid',target_centroid)

            source_centering_x=source_x-source_centroid[0]
            source_centering_y=source_y-source_centroid[1]

            target_centering_x=target_x-target_centroid[0]
            target_centering_y=target_y-target_centroid[1]

            panning_x=target_centroid[0]
            panning_y=target_centroid[1]
            # print('平移中心點')

            if show:
                show_plt(target_centering_x, target_centering_y)
                show_plt(source_centering_x, source_centering_y)
                plt.show()

            #第一次旋轉################################################################################################################################################
            source_centering = np.vstack((source_centering_x, source_centering_y)).T
            target_centering = np.vstack((target_centering_x,target_centering_y)).T

            source_angle_degrees = LR_degree(source_centering)
            target_angle_degrees = LR_degree(target_centering)

            rotate_1=source_angle_degrees-target_angle_degrees
            # print('第一次旋轉 : ',rotate_1,'度')
            source_panning_rotation_final = target_change(source_centering, zoom_x=1, zoom_y=1, translate_x=0,
                                                                               translate_y=0, rotate=rotate_1)


            source_final_x = source_panning_rotation_final[:, 0]
            source_final_y = source_panning_rotation_final[:, 1]

            target_centering_x = target_centering[:, 0]
            target_centering_y = target_centering[:, 1]

            if show:
                target_centroid = np.mean(target_centering, axis=0)
                source_centroid = np.mean(source_panning_rotation_final, axis=0)


                plt.scatter(source_centroid[0], source_centroid[1], color='red', marker='x', label='Centroid')
                plt.scatter(target_centroid[0], target_centroid[1], color='blue', marker='x', label='Centroid')
                show_plt(target_centering_x, target_centering_y)
                show_plt(source_final_x, source_final_y)
                plt.show()

            source_final = np.vstack((source_final_x,source_final_y)).T


            #Gaussian processes################################################################################################################################################
            # res = gp_minimize(fitness_fun,     # the function to minimize
            #           [(0.95, 1.0),(0.95, 1.0),(-1.0, 1.0)],    # the bounds on each dimension of x
            #           acq_func="PI",    # the acquisition function
            #           n_calls=10,      # the number of evaluations of f
            #           n_random_starts=5,  # the number of random initialization points
            #           )

            # print(res['x'])
            # source_gaussian = target_change(source_final, zoom_x=res['x'][0], zoom_y=res['x'][1], translate_x=0,
            #                        translate_y=0, rotate=res['x'][2])
            source_gaussian=source_final

            source_gaussian_x = source_gaussian[:, 0]
            source_gaussian_y = source_gaussian[:, 1]
            #
            # if show:
            #     show_plt(target_centering_x, target_centering_y)
            #     show_plt(source_gaussian_x, source_gaussian_y)
            #     plt.show()

            # Gaussian processes ################################################################################################################################################
            source_gaussian_x=source_gaussian_x+panning_x
            source_gaussian_y=source_gaussian_y+panning_y



            source_gaussian = np.vstack((source_gaussian_x, source_gaussian_y)).T


            source_gaussian_x = source_gaussian[:, 0]
            source_gaussian_y = source_gaussian[:, 1]
            if show:
                show_plt(target_x, target_y)
                show_plt(source_gaussian_x, source_gaussian_y)
                # plt.show()


            distances = cdist(source_gaussian, target_np)
            # print(distances)
            min_index = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
            # print(min_index[0],min_index[1])
            if show:
                show_plt(target_np[min_index[1]][0],target_np[min_index[1]][1])
                show_plt(source_gaussian[min_index[0]][0], source_gaussian[min_index[0]][1])
                # plt.show()
                # print('find min')

            # print(target_np)
            source_fit,test_dis,source_index,target_index=fit_to_target(source_gaussian,target_np,closest_source=min_index[0],closest_target=min_index[1])#
            np.save("last_point.npy",source_fit)
        fit_dis.append(test_dis)
        source_inds.append(source_index)
        target_inds.append(target_index)




        FIT_piont_list.append(source_fit)
    tt=fit_dis.index(min(fit_dis))
    # print(fit_dis, 'wwwwww')
    # print(source_inds, 'wwwwww')
    # print(try_angle[tt],fit_dis[tt])
    if show_f :
        xs = target_change(source_npb, zoom_x=1.0, zoom_y=1.0, translate_x=0, translate_y=0, rotate=try_angle[tt])
        FIT_piont, test_dis = Fit_show(xs, ys, show=1)
        FIT_piont_x = FIT_piont[:, 0]
        FIT_piont_y = FIT_piont[:, 1]
        show_plt(FIT_piont_x, FIT_piont_y)
        print(test_dis)
        for i in range(np.shape(FIT_piont)[0]):
            plt.text(FIT_piont[i, 0], FIT_piont[i, 1], f'{i}', fontsize=30, ha='left')
    # newindex=[]
    # for i in range (len(source_inds[tt])):
    #     source_inds[tt].index(i)
    #     newindex.append(target_inds[tt][source_inds[tt].index(i)])
    # print(newindex)
    # sort_fit=[]
    # for i in range(len(newindex)):
    #     sort_fit.append(list(FIT_piont_list[tt][newindex.index(i)]))
    # print(sort_fit)
    # return sort_fit,fit_dis[tt],try_angle[tt],source_inds[tt]
    return FIT_piont_list[tt],fit_dis[tt]





# xs = np.array([[0.93578, 0.74444],[0.97578, 0.83444], [0.92202, 0.15], [0.06422, 0.84444], [0.27064, 0.82778], [0.70872, 0.78333], [0.49541, 0.8], [0.48165, 0.21111], [0.70642, 0.17222], ])
# xs = np.array([[0.264 ,0.155 ],[0.638 ,0.071 ],[0.362 ,0.300 ],[0.610 ,0.261 ],[0.930 ,0.507 ],[0.580 ,0.678 ],[0.265 ,0.475 ],[0.728 ,0.452 ],[0.064 ,0.421 ],[0.380 ,0.685 ],[0.326 ,0.914 ],[0.688 ,0.863 ]])

# # ys = np.array([[0.43023, 0.62295], [0.065116, 0.31148], [0.61628, 0.78142], [0.78605, 0.93169], [0.24884, 0.46995],[0.76279, 0.53552], [0.93953, 0.69126], [0.78605, 0.99169], [0.5814, 0.37432]])
# ys = np.array([[0.264, 0.155], [0.638, 0.071], [0.362, 0.300], [0.610, 0.261], [0.930, 0.507], [0.580, 0.678], [0.265, 0.475],[0.728, 0.452], [0.064, 0.421], [0.380, 0.685], [0.326, 0.914], [0.688, 0.863]])
# xsb = xs
# init_rotate = -80
# # print(len(ys))
# try_angle = list(range(0,360,int(360/len(xs))))

# #
# # for i in try_angle:#[10,20,30,40,50,60,70,80,90,100,110,120,130,140]:
# #     print(i,'旋轉測試')

# #     ys = target_change(ysb, zoom_x=1.0, zoom_y=1.0, translate_x=0, translate_y=0, rotate=init_rotate)

# #     ys = target_change(ys, zoom_x=1.0, zoom_y=1.0, translate_x=0,translate_y=0, rotate=i)
# #     FIT_piont ,test_dis =Fit(xs,ys,show=0)
# #     FIT_piont_x = FIT_piont[:, 0]
# #     FIT_piont_y = FIT_piont[:, 1]
# #     #show_plt(FIT_piont_x, FIT_piont_y)

# #     fit_dis.append(test_dis)
# #     FIT_piont_list.append(FIT_piont)
# # print(fit_dis)
# # tt=fit_dis.index(min(fit_dis))
# # print(try_angle)
# # print(tt)
# # print(try_angle[tt],fit_dis[tt])
# # show_plt(FIT_piont_list[tt][:, 0], FIT_piont_list[tt][:, 1])


# xs = target_change(xsb, zoom_x=1.0, zoom_y=1.0, translate_x=0, translate_y=0, rotate=init_rotate)
# # ys = target_change(ys, zoom_x=1.0, zoom_y=1.0, translate_x=0,translate_y=0, rotate=try_angle[tt])

# FIT_piont,test_dis,rotate_angle=Fit(xs,ys,show=0,show_f=1)
# FIT_piont_x = FIT_piont[:, 0]
# FIT_piont_y = FIT_piont[:, 1]
# show_plt(FIT_piont_x, FIT_piont_y)
# print(test_dis)
# # for i in range(np.shape(FIT_piont)[0]):
# #     plt.text(FIT_piont[i, 0], FIT_piont[i, 1], f'{i}', fontsize=30, ha='left')

# ys_x = ys[:, 0]
# ys_y = ys[:, 1]
# # print(rotate_angle)
# # ys = target_change(ysb, zoom_x=1.0, zoom_y=1.0, translate_x=0, translate_y=0, rotate=init_rotate)
# # ys = target_change(ys, zoom_x=1.0, zoom_y=1.0, translate_x=0,translate_y=0, rotate=rotate_angle)
# # FIT_piont,test_dis=Fit_show(xs,ys,show=1)

# # show_plt(ys_x, ys_y)
# # for i in range(np.shape(ys)[0]):
# #     plt.text(ys[i, 0],ys[i, 1], f'{i}', fontsize=30, ha='right')

# plt.show()

# [[    0.47832   ,  0.26763],[    0.47416   , 0.092931],[    0.70701   ,  0.36918],[    0.83378   ,  0.22624],[    0.70306   ,  0.59471],[    0.83935   ,  0.75179],[    0.45168   ,  0.70168],[     0.4418  ,  0.91617],[    0.28458  ,   0.60093],[    0.11248   , 0.76205],[    0.30226   ,  0.36167],[    0.16291    ,  0.2038]]
# [     0.4853    , 0.23873] ,[    0.48387  ,  0.063989], [    0.71236  ,   0.34385], [    0.84136  ,   0.20292], [    0.70487  ,    0.5693], [    0.83869  ,   0.72849], [    0.45185  ,   0.67231], [    0.43861  ,   0.88662], [    0.28635  ,   0.56895], [    0.11175  ,   0.72736], [    0.30778  ,      0.33], [    0.17092 ,    0.16996]