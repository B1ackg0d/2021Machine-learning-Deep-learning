import numpy as np
import struct
from sklearn.decomposition import PCA
from sklearn import tree
# from sklearn.neighbors import KDTree#导入KD树类
from keras.datasets import mnist
import matplotlib.pyplot as plt
def visualize(X,y):
    '嵌入空间可视化'''
    x_min, x_max = X.min(0), X.max(0)
    X_norm = (X - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.show()

if __name__ == "__main__":
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
    pca = PCA(n_components = 3)
    pca.fit(train_data) #fit PCA with training data instead of the whole dataset
    train_data_pca = pca.transform(train_data)
    test_data_pca = pca.transform(test_data)
    print("PCA completed with 100 components")
    print ("training data shape after PCA:",train_data_pca.shape)
    print ("testing data shape after PCA:",test_data_pca.shape)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    train_data_pca=train_data_pca[0:1000]
    train_label=train_label[0:1000]
    ax.scatter(train_data_pca[:, 0], train_data_pca[:, 1], train_data_pca[:, 2], c=train_label, cmap=plt.cm.Spectral)
    plt.show()
    # visualize(train_data_pca,train_label)
    # train_num = 60000
    # test_num = 10000
    # clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='best',min_samples_split=6,min_samples_leaf=3)
    # clf.fit(train_data_pca[:train_num], train_label[:train_num])
    # # clf.fit(train_data_pca, train_label)
    # # 预测
    # prediction = clf.predict(test_data_pca)
    # accurancy = clf.score(test_data_pca,test_label)
    # # accurancy = np.sum(np.equal(prediction, test_label)) / test_num
    # print('prediction : ', prediction)
    # print('accurancy : ', accurancy)