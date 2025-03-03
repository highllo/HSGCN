import torch
import tqdm
from HSI_dataloader import Mydataset
from torch import load
from torch_geometric.data import DataLoader
from module.utils import *
from module.utils.parser import parse_args
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score, precision_score, f1_score,average_precision_score,cohen_kappa_score
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CPU = torch.device( "cpu")
#device = torch.device("cpu")
if __name__ == '__main__':


    set_seed(19930819)
    args = parse_args()
    dataset_name = 'HSI'
    _hidden_size = 256
    _num_labels = 12
    debias_flag = False
    topN = None
    batch_size = 1
    scope = 'part'
    path = 'D:\PhD\science\project\\featureattr\\reinforced_causal_explainer-master\model\Re_256_GCN.pt'
    test_dataset = Mydataset('Data/OHSC', mode = 'testing', k1 =256)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = torch.load(path)
    model.eval()

    Y_true = torch.tensor([]).to(device)
    Y_pred = torch.tensor([]).to(device)
    for graph in iter(test_loader):
        graph = graph.to(device)
        graph_pred = model(graph.x,graph.edge_index,graph.edge_attr,graph.batch)

        pred = graph_pred.argmax(dim=1)
        Y_true = torch.cat((Y_true, graph.y), dim=0)
        Y_pred = torch.cat((Y_pred, pred), dim=0)

    Y_true = Y_true.to(CPU)
    Y_pred = Y_pred.to(CPU)
    acc = accuracy_score(Y_true, Y_pred)
    kappa = cohen_kappa_score(Y_true, Y_pred)
    print('ACC: ',  acc)
    print('kappa: ', kappa)
