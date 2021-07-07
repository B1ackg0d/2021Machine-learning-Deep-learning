import numpy as np
import pandas as pd
from collections import Counter
import time
from keras.datasets import mnist
from sklearn.decomposition import PCA

allow_duplicate = False


class Ball():
    def __init__(self,center,radius,points,left,right):
        self.center = center      #使用该点即为球中心,而不去精确地去找最小外包圆的中心
        self.radius = radius
        self.left = left
        self.right = right
        self.points = points

class BallTree():
    def __init__(self,values,labels):
        self.values = values
        self.labels = labels
        if(len(self.values) == 0 ):
            raise Exception('Data For Ball-Tree Must Be Not empty.')
        self.root = self.build_BallTree()
        self.KNN_max_now_dist = np.inf
        self.KNN_result = [(None,self.KNN_max_now_dist)]

    def build_BallTree(self):
        data = np.column_stack((self.values,self.labels))
        return self.build_BallTree_core(data)

    def dist(self,point1,point2):
        return np.sqrt(np.sum((point1-point2)**2))

    #data:带标签的数据且已经排好序的
    def build_BallTree_core(self,data):
        if len(data) == 0:
            return None
        if len(data) == 1:
            return Ball(data[0,:-1],0.001,data,None,None)
        #当每个数据点完全一样时,全部归为一个球,及时退出递归,不然会导致递归层数太深出现程序崩溃
        data_disloc =  np.row_stack((data[1:],data[0]))
        if np.sum(data_disloc-data) == 0:
            return Ball(data[0, :-1], 1e-100, data, None, None)
        cur_center = np.mean(data[:,:-1],axis=0)     #当前球的中心
        print(data[:,:-1].shape)
        print(cur_center.shape)
        print(cur_center)
        dists_with_center = np.array([self.dist(cur_center,point) for point in data[:,:-1]])     #当前数据点到球中心的距离
        max_dist_index = np.argmax(dists_with_center)        #取距离中心最远的点,为生成下一级两个子球做准备,同时这也是当前球的半径
        max_dist = dists_with_center[max_dist_index]
        root = Ball(cur_center,max_dist,data,None,None)
        point1 = data[max_dist_index]

        dists_with_point1 = np.array([self.dist(point1[:-1],point) for point in data[:,:-1]])
        max_dist_index2 = np.argmax(dists_with_point1)
        point2 = data[max_dist_index2]            #取距离point1最远的点,至此,为寻找下一级的两个子球的准备工作搞定

        dists_with_point2 = np.array([self.dist(point2[:-1], point) for point in data[:, :-1]])
        assign_point1 = dists_with_point1 < dists_with_point2

        root.left = self.build_BallTree_core(data[assign_point1])
        root.right = self.build_BallTree_core(data[~assign_point1])
        return root    #是一个Ball

    def search_KNN(self,target,K):
        if self.root is None:
            raise Exception('KD-Tree Must Be Not empty.')
        if K > len(self.values):
            raise ValueError("K in KNN Must Be Greater Than Lenght of data")
        if len(target) !=len(self.root.center):
            raise ValueError("Target Must Has Same Dimension With Data")
        self.KNN_result = [(None,self.KNN_max_now_dist)]
        self.nums = 0
        self.search_KNN_core(self.root,target,K)
        return self.nums
        # print("calu_dist_nums:",self.nums)

    def insert(self,root_ball,target,K):
        for node in root_ball.points:
            self.nums += 1
            is_duplicate = [self.dist(node[:-1], item[0][:-1]) < 1e-4 and
                            abs(node[-1] - item[0][-1]) < 1e-4 for item in self.KNN_result if item[0] is not None]
            if np.array(is_duplicate, np.bool).any() and not allow_duplicate:
                continue
            distance = self.dist(target,node[:-1])
            if(len(self.KNN_result)<K):
                self.KNN_result.append((node,distance))
            elif distance < self.KNN_result[0][1]:
                self.KNN_result = self.KNN_result[1:] + [(node, distance)]
            self.KNN_result = sorted(self.KNN_result, key=lambda x: -x[1])


    #root是一个Ball
    def search_KNN_core(self,root_ball, target, K):
        if root_ball is None:
            return
        #在合格的超体空间(必须是最后一层的子空间)内查找更近的数据点
        if root_ball.left is None or root_ball.right is None:
            self.insert(root_ball, target, K)
        if abs(self.dist(root_ball.center,target)) <= root_ball.radius + self.KNN_result[0][1] : #or len(self.KNN_result) < K
            self.search_KNN_core(root_ball.left,target,K)
            self.search_KNN_core(root_ball.right,target,K)


if __name__ == '__main__':
    (train_data_ori, train_label), (test_data_ori, test_label) = mnist.load_data()
    train_data_ori= train_data_ori.astype('float32')
    test_data_ori= test_data_ori.astype('float32')
    train_data_ori/= 255
    test_data_ori /= 255
    print ("mnist data loaded")
    print ("original training data shape:",train_data_ori.shape)
    print ("original testing data shape:",test_data_ori.shape)
    train_data=train_data_ori.reshape(60000,784)
    test_data=test_data_ori.reshape(10000,784)
    print ("training data shape after reshape:",train_data.shape)
    print ("testing data shape after reshape:",test_data.shape)
    pca = PCA(n_components = 100)
    pca.fit(train_data) #fit PCA with training data instead of the whole dataset
    train_data_pca = pca.transform(train_data)
    test_data_pca = pca.transform(test_data)
    # csv_path = "winequality-white.csv"
    # data,lables,dim_label = load_data(csv_path)
    # split_rate = 0.8 ;
    K=7

    start1 = time.time()
    # ball_tree = BallTree(data[:train_num], lables[:train_num])
    ball_tree = BallTree(train_data_pca , train_label)
    end1 = time.time()

    diff_all=0
    accuracy = 0
    search_all_time = 0
    calu_dist_nums = 0
    for index,target in enumerate(test_data_pca):
        start2 = time.time()
        calu_dist_nums+=ball_tree.search_KNN(target, K)
        end2 = time.time()
        search_all_time += end2 - start2

        # for res in ball_tree.KNN_result:
        #     print("res:",res[0][:-1],res[0][-1],res[1])
        pred_label = Counter(node[0][-1] for node in ball_tree.KNN_result).most_common(1)[0][0]
        diff_all += abs(test_label[index] - pred_label)
        if (test_label[index] - pred_label) == 0:
            accuracy += 1
        print("accuracy:", accuracy / (index + 1))
        print("Total:{},MSE:{:.3f}    {}--->{}".format(index + 1, (diff_all / (index + 1)), test_label[index],
                                                   pred_label))


    print("BallTree构建时间：", end1 - start1)
    print("程序运行时间：", search_all_time/10000)
    print("平均计算次数：", calu_dist_nums /10000)