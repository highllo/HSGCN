import torch
from torch_geometric.data import DataLoader
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
from torch_geometric.utils import subgraph as sb
import os
import random
import math
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix
import copy
from abc import ABC
from typing import Any
import numpy as np
import scipy.sparse as sp


def LargestConnectedComponents(data,connection = 'strong',num_comp=1):
    num_nodes = data.num_nodes
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes= num_nodes)
    num_components, component = sp.csgraph.connected_components(adj, connection=connection)
    _, count = np.unique(component, return_counts=True)
    subset_np = np.in1d(component, count.argsort()[-num_components:])
    subset = torch.from_numpy(subset_np)
    subset = subset.to(data.edge_index.device, torch.bool)
    edge_index, edge_attr = sb(subset, data.edge_index)
    data.edge_attr = torch.ones(edge_index.shape[1],1)
    sub_nodes = torch.unique(edge_index)
    data.x = data.x[sub_nodes]
    if num_components <= num_comp:
        return data
    if data.x.shape[0] == 0:
        print('error')
        f = 'error'
        return f
    data.batch = data.batch[sub_nodes]
    row, col = edge_index
        # remapping the nodes in the explanatory subgraph to new ids.
    node_idx = row.new_full((num_nodes,), -1)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
    data.edge_index = node_idx[edge_index]
    return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.num_components})'

def relabel_graph(graph, selection):
    subgraph = copy.deepcopy(graph)

    # retrieval properties of the explanatory subgraph
    # .... the edge_index.
    subgraph.edge_index = graph.edge_index.T[selection].T
    # .... the edge_attr.
    subgraph.edge_attr = graph.edge_attr[selection]
    # .... the nodes.
    sub_nodes = torch.unique(subgraph.edge_index)
    # .... the node features.
    subgraph.x = graph.x[sub_nodes]
    subgraph.batch = graph.batch[sub_nodes]

    row, col = graph.edge_index
    # remapping the nodes in the explanatory subgraph to new ids.
    node_idx = row.new_full((graph.num_nodes,), -1)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
    subgraph.edge_index = node_idx[subgraph.edge_index]

    return subgraph

def euclidean_distance(vector1, vector2):

    distance = 0.0
    for i in range(len(vector1)):
        distance += (vector1[i] - vector2[i]) ** 2
    return math.sqrt(distance)

import numpy as np


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

        assert mode in self.splits##检查设置的模式是否在划分的两个类别中
        self.mode = mode
        self.k = k1

        super(Mydataset, self ).__init__(root, transform, pre_transform, pre_filter)

        idx = self.processed_file_names.index('{}_{}.pt'.format(mode,self.k))  ##index(a) 是查找列表中的对应元素 a 的第一个位置索引值 0 1 2
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
                                           #'tif')  # 已经标准化的高光谱图片
        #image_data = dataset.MyImageFolder(r'E:\DATASET\splitdata\hscohs-tif\test', 'shsc',
                                           #'tif')  # 已经标准化的高光谱图片

        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        model = torch.load(
            'model\Re_{}_GCN.pt'.format(k))

        test_dataset = Mydataset('Data/HSC', mode='testing', k1=k)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        for graph in tqdm(iter(test_loader)):
            graph = graph.to(device)
            #out = model(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
            #out = F.softmax(out)

            # 输入图graph 输出最优子图subgraph 包含所有节点，但边最少
            path_e = 'HSI_0.0001_1e-0516v1.pt'
            rc_explainer = torch.load(path_e, map_location=lambda storage, loc: storage.cuda(0))  ##加载训练好的边筛选器
            rc_explainer.eval()
            model.eval()
            # topK_ratio_list = [0.1]
            max_budget = graph.num_edges
            state = torch.zeros(max_budget, dtype=torch.bool)
            # check_budget_list = [max(int(_topK * max_budget), 1) for _topK in topK_ratio_list]
            valid_budget = max(int(max_budget), 1)
            # imp = list()
            for budget in range(valid_budget):
                available_actions = state[~state].clone()
                _, _, make_action_id, _ = rc_explainer(graph=graph, state=state, train_flag=False)
                available_actions[make_action_id] = True
                state[~state] = available_actions.clone()
                sub_temp = relabel_graph(graph, state)
                rows = sub_temp.x.shape[0]
                if rows == k:
                    subgraph0 = relabel_graph(graph, state)
                    break
            subgraph = LargestConnectedComponents(subgraph0)
            data = Data(x=subgraph.x, y=subgraph.y, edge_index=subgraph.edge_index, edge_attr=subgraph.edge_attr)
            data_list_test.append(data)
        random.shuffle(data_list_test)
        torch.save(self.collate(data_list_test),
                   'Data\\SHSC\processed\\testing_{}.pt'.format(
                       self.k))


        train_dataset = Mydataset('Data/HSC', mode='training', k1=k)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        for graph in tqdm(iter(train_loader)):
            graph = graph.to(device)

            # 输入图graph 输出最优子图subgraph 包含所有节点，但边最少
            #path_e = 'HSI_0.0001_1e-05{}.pt'.format(k)
            path_e = 'HSI_0.0001_1e-0516v1.pt'
            rc_explainer = torch.load(path_e, map_location=lambda storage, loc: storage.cuda(0))  ##加载训练好的边筛选器
            rc_explainer.eval()
            model.eval()
            max_budget = graph.num_edges
            state = torch.zeros(max_budget, dtype=torch.bool)
            valid_budget = max(int(max_budget), 1)
            for budget in range(valid_budget):
                available_actions = state[~state].clone()
                _, _, make_action_id, _ = rc_explainer(graph=graph, state=state, train_flag=False)
                available_actions[make_action_id] = True
                state[~state] = available_actions.clone()
                sub_temp = relabel_graph(graph, state)
                rows = sub_temp.x.shape[0]
                if rows == k:
                    subgraph0 = relabel_graph(graph, state)
                    break

            subgraph = LargestConnectedComponents(subgraph0)
            data_1 = Data(x=subgraph.x, y=subgraph.y, edge_index=subgraph.edge_index, edge_attr=subgraph.edge_attr)
            data_list_train.append(data_1)
        random.shuffle(data_list_train)
        torch.save(self.collate(data_list_train),
                   'Data\\SHSC\processed\\training_{}.pt'.format(
                       self.k))

if __name__ == '__main__':
    for k in [16,32,64,128,256]:
        test_dataset = Mydataset('Data\\SHSC', mode='testing' , k1 = k)
        training_dataset = Mydataset('Data\\SHSC', mode='training', k1=k)
