import torch
from torch.utils.data import DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from sklearn.cluster import KMeans
from tqdm import tqdm
import dataset
import numpy as np
import os
import random
import math


def euclidean_distance(vector1, vector2):
    distance = 0.0
    for i in range(len(vector1)):
        distance += (vector1[i] - vector2[i]) ** 2
    return math.sqrt(distance)



def euclidean_dist(x, y):
    x = np.array(x)
    y= np.array(y)
    m, n = x.shape[0], y.shape[0]
    xx = np.expand_dims(np.sum((x ** 2), axis=1), axis=1).repeat(n, axis=1)
    yy = np.expand_dims(np.sum((y ** 2), axis=1), axis=1).repeat(m, axis=1).T
    dist = xx + yy - 2 * x @ y.T
    dist = np.sqrt(np.clip(dist, a_min=1e-12, a_max=None))
    return dist.T


def knn(data_support, center_data, k, num):

    dist = euclidean_dist(data_support, center_data)
    sorted_idxs = dist.argsort(axis=1)
    lst = dist.tolist()
    lst = np.array(lst)
    lst.T[np.lexsort(lst[::-1, :])].T
    sorted_1 = lst.sort()
    a,_ = data_support.shape
    connect_idx =  sorted_idxs[0,1:k+1]
    dist1 = lst [0, 1:k + 1]
    return connect_idx,dist1

class Mydataset(InMemoryDataset):

    splits = ['training', 'testing']

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, mode='testing', k1 = 5):
        assert mode in self.splits
        self.mode = mode
        self.k = k1
        super(Mydataset, self ).__init__(root, transform, pre_transform, pre_filter)
        idx = self.processed_file_names.index('{}_{}.pt'.format(mode,self.k))
        print('{}_{}.pt'.format(mode,self.k))
        self.data,self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)##raw_file_names包含目录下所有的文件名


    @property
    def processed_file_names(self):
        #return ['training.pt', 'evaluation.pt', 'testing.pt']
        return ['training_{}.pt'.format(self.k), 'testing_{}.pt'.format(self.k)]

    def download(self):
        pass

    def process(self):
        data_list_train = []#保存最终生成图的结果
        data_list_test = []  # 保存最终生成图的结果
        #image_data = dataset.MyImageFolder(r'E:\HSIdata\splitdata\hsi\test', 'shsc',
                                           #'tif')
        #image_data = dataset.MyImageFolder(r'E:\DATASET\splitdata\hscohs-tif\test', 'shsc',
                                           #'tif')
        image_data = dataset.MyImageFolder(r'E:\DATASET\splitdata\OHS-tif\test', 'ohs',
                                           'tif')  # 已经标准化的高光谱图片
        pic_test = DataLoader(image_data, batch_size=1, shuffle=None)
        cluster = KMeans(n_clusters=self.k, init='k-means++', n_init=50, max_iter=500,
                         tol=0.0001)
        for i, data_ in tqdm(enumerate(pic_test)):
            img_test = torch.squeeze(data_[0],0)
            #plt.imshow(img_test)
            #plt.show()
            label = data_[1]
            height, width, bands = img_test.shape
            sample_test = np.reshape(img_test, [height * width, bands]).numpy()
            cluster.fit(sample_test)
            cernter_test = cluster.cluster_centers_
            feature = torch.tensor(cernter_test)
            node_label = torch.tensor(np.arange(0,self.k), dtype=torch.float)
            knn_num = 4
            source = np.zeros((1, feature.shape[0] * knn_num))
            target = np.zeros((1, feature.shape[0] * knn_num))
            eaddr = np.zeros((feature.shape[0] * knn_num, 1), dtype=float)
            for j in range(feature.shape[0]):
                source[0, j * knn_num:(j + 1) * knn_num] = [j]
            for j in range(feature.shape[0]):
                center = feature[j]
                center = center.reshape((1,feature.shape[1]))
                connect_idx,dist = knn(feature, center, knn_num, j)
                target[0, j * knn_num:(j + 1) * knn_num] = connect_idx
                eaddr[j * knn_num:(j + 1) * knn_num, 0] = dist[0:knn_num]
            eattr = torch.tensor(eaddr).float()
            source_node = torch.LongTensor(source)
            target_node = torch.LongTensor(target)
            edge_index = torch.cat([source_node, target_node], 0)
            data = Data(x=feature, y=label, edge_index=edge_index,edge_attr=eattr)
            data_list_test.append(data)
        random.shuffle(data_list_test)
        torch.save(self.collate(data_list_test),
                   'Data\\OHS\processed\\testing_{}.pt'.format(
                       self.k))

        ####read trainset
        image_data1 = dataset.MyImageFolder(r'E:\DATASET\splitdata\OHS-tif\train', 'ohs',
                                           'tif')  # 已经标准化的高光谱图片
        pic_train = DataLoader(image_data1, batch_size=1, shuffle=None)
        for i, data_ in tqdm(enumerate(pic_train)):
            img_train = torch.squeeze(data_[0],0)
            label_1 = data_[1]
            height, width, bands = img_train.shape
            sample_train = np.reshape(img_train, [height * width, bands]).numpy()
            cluster.fit(sample_train)
            cernter_train = cluster.cluster_centers_
             # 输出分类结果，转换成tensor格式
            feature_1 = torch.tensor(cernter_train)
            # 构建边矩阵 2行90列 （knn）
            knn_num = 4
            source_1 = np.zeros((1, feature_1.shape[0] * knn_num))
            target_1 = np.zeros((1, feature_1.shape[0] * knn_num))
            eaddr_1 = np.zeros((feature_1.shape[0] * knn_num, 1), dtype=float)


            for j in range(feature_1.shape[0]):
                source_1[0, j * knn_num:(j + 1) * knn_num] = [j]
            for j in range(feature_1.shape[0]):
                center = feature_1[j]
                center = center.reshape((1,feature_1.shape[1]))
                connect_idx,dist = knn(feature_1, center, knn_num, j)
                target_1[0, j * knn_num:(j + 1) * knn_num] = connect_idx
                eaddr_1[j * knn_num:(j + 1) * knn_num, 0] = dist[0:knn_num]

            eattr_1 = torch.tensor(eaddr_1).float()
            source_node = torch.LongTensor(source_1)
            target_node = torch.LongTensor(target_1)
            edge_index_1 = torch.cat([source_node, target_node], 0)
            data_1 = Data(x=feature_1, y=label_1, edge_index=edge_index_1,edge_attr=eattr_1)
            data_list_train.append(data_1)
        random.shuffle(data_list_train)
        data_2, slices = self.collate(data_list_train)

        #torch.save(self.collate(data_list[:1267]), 'Data\HSI\processed\\testing_{}.pt'.format(self.k))
        #torch.save(self.collate(data_list[1268:6371]),'Data\HSI\processed\\training_{}.pt'.format(self.k))

        torch.save(self.collate(data_list_train),'Data\\OHSV1\processed\\training_{}.pt'.format(self.k))

if __name__ == '__main__':
    for k in [256]:
    #for k in [16, 32, 64, 128, 256, 512]:
        train_dataset = Mydataset('Data\\OHSV1', mode='training', k1=k)
        test_dataset = Mydataset('Data\\OHSV1', mode='testing', k1=k)