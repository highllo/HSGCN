from torch.nn import Sequential as Seq, ReLU, Tanh, Linear as Lin, Softmax
import torch
from torch_geometric.nn import GCNConv, APPNP, BatchNorm, global_mean_pool,models
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool,BatchNorm

class GCN(torch.nn.Module):
    def __init__(self, channel, hidden1, hidden2,hidden3, classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(channel, hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)
        self.conv3 = GCNConv(hidden2, hidden3)
        #self.conv4 = GCNConv(hidden3, hidden4)
        #self.lin = Linear(6, dataset.num_classes)
        self.norm = BatchNorm(hidden3)
        #self.norm4 = BatchNorm(hidden4)
        self.softmax = Softmax(dim=1)
        self.mlp = nn.Sequential(OrderedDict([
            ('lin1', Lin(hidden3, 64)),
            ('lin2', Lin(64, classes)),
            #('lin3', Lin(512, classes)),
        ]))

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.get_graph_rep(x, edge_index, edge_attr, batch)
        return self.get_pred(x)

    def get_pred(self, graph_x):
        pred = self.mlp(graph_x)
        self.readout = self.softmax(pred)
        return pred

    def get_node_reps(self, x,edge_index,edge_attr,batch):
        edge_weight = edge_attr.view(-1)
        x = F.relu(self.conv1(x, edge_index,edge_weight))
        #x = self.norm1(x)
        x = F.relu(self.conv2(x, edge_index,edge_weight))
        #x = self.norm2(x)
        node_x = F.relu(self.conv3(x, edge_index, edge_weight))
        #node_x = F.relu(self.conv4(x, edge_index,edge_weight))
        #x = self.norm4(x)
        node_x = self.norm(node_x)
        return node_x

    def get_graph_rep(self, x,edge_index,edge_attr,batch):
        node_x = self.get_node_reps(x,edge_index,edge_attr,batch)
        graph_x = global_mean_pool(node_x, batch)
        return graph_x



