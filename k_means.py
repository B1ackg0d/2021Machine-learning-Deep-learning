import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset,Dataset
from torchvision.datasets import DatasetFolder
import matplotlib.pyplot as plt
from kmeans_pytorch import kmeans, kmeans_predict
from tqdm.auto import tqdm
train_tfm = transforms.Compose([
    transforms.Resize((128,128)),
    # transforms.RandomHorizontalFlip(),#随机水平翻转
    # transforms.RandomRotation(15),
    # transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])


test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])

if __name__ == '__main__':
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    train_set = DatasetFolder("food-11/training/labeled", loader=lambda x: Image.open(x), extensions="jpg",
                              transform=train_tfm)
    valid_set = DatasetFolder("food-11/validation", loader=lambda x: Image.open(x), extensions="jpg",
                              transform=test_tfm)
    # unlabeled_set = DatasetFolder("food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg",
    #                               transform=train_tfm)
    # test_set = DatasetFolder("food-11/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
    #
    train_loader = DataLoader(train_set,  batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    # valid_loader = DataLoader(valid_set, shuffle=True, num_workers=0, pin_memory=True)
    # test_loader = DataLoader(test_set,  shuffle=False)
    num_clusters=11

    # data=train_set.datas
    data_size, dims, num_clusters = 50, 2, 3
    x = np.random.randn(data_size, dims) / 6
    x = torch.from_numpy(x)
    print(x[1])
    dataSet = np.zeros((len(train_loader), 1*3*128*128))
    i=0
    tensor_list = list()
    ss= train_set[1][0].reshape(1*3*128*128)
    print(ss.shape)
    for i in range(len(train_set)):
        # print(train_set[i][0].reshape(1*3*128*128))
        ss=train_set[i][0].reshape(1*3*128*128)
        tensor_list.append(ss)
    # for batch in tqdm(train_loader):
    #         img,_ = batch
    #         # arr=img.numpy()
    #         # print(arr.shape)
    #         # arr.reshape(1*3*128*128)
    #         img.reshape(1*3*128*128)
    #         tensor_list.append(img)
    #
    #         # dataSet[i][:]= dataSet[i][:] + arr
    #         # i+=1
    final_tensor = torch.stack(tensor_list)
    print(final_tensor.shape)
    cluster_ids_x, cluster_centers = kmeans(
        X=final_tensor , num_clusters=num_clusters, distance='euclidean', device='cuda'
    )
    print(cluster_centers)