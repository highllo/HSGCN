import numpy as np
from  torch_geometric.data import DataLoader
import  torch.nn.functional as  F
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score, precision_score, f1_score,average_precision_score,cohen_kappa_score


from HSI_dataloader import Mydataset
from net import GCN
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
CPU = torch.device("cpu")
dataset_name = 'HSC'
learning_rate = 0.0001
max_epoch = 500

def plot_confusion_matrix(cm, labels_name, title, colorbar=False, cmap=None):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    if colorbar:
        plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name,rotation = 30,fontsize = 6)
    plt.yticks(num_local, labels_name)
    plt.title(title)    # 图像标题
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

#for k in [16]:
for k in [256]:
    recall = 0
    precision = 0
    acc = 0
    f1 = 0
    DATASET = 'HSC'
    hidden1 = 256
    hidden2 = 512
    hidden3 = 256
    #hidden4 = 128
    classes = 12
    channel = 32
    PATH = str(dataset_name) + "_" + str(k) + "_GCNv2.pt"
    train_dataset = Mydataset('Data\\'+DATASET, mode='training', k1 = k)
    #train_dataset = Mydataset('Data\\nasc_tg2', mode='training', k1 = k)
    test_dataset = Mydataset('Data\\'+DATASET, mode='testing' , k1 = k)
    #test_dataset = Mydataset('Data\\nasc_tg2', mode='testing', k1=k)
    def train():
        model.train()
        loss_all = 0
        for graph in train_loader:
            graph = graph.to(device)
            optimizer.zero_grad()
            output = model(graph.x, graph.edge_index,graph.edge_attr,graph.batch)
            label = graph.y
            loss = crit(output, label)
            loss.backward()
            loss_all += graph.num_graphs * loss.item()
            optimizer.step()
        return loss_all / len(train_dataset)
    model = GCN(channel, hidden1, hidden2, hidden3,classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    crit = torch.nn.CrossEntropyLoss()
    train_loader = DataLoader(train_dataset, batch_size=96)
    test_loader = DataLoader(test_dataset, batch_size=96)

    def test(loader,mode = 'train'):
         model.eval()
         correct=0
         Y_true =  torch.tensor([]).to(device)
         Y_pred = torch.tensor([]).to(device)
         if mode == 'train':
            for graph in loader:
                graph = graph.to(device)
                out = model(graph.x, graph.edge_index,graph.edge_attr,graph.batch)
                out = F.softmax(out)
                pred = out.argmax(dim=1)
                correct += int((pred==graph.y).sum())
            return correct/len(loader.dataset)

         elif mode == 'test1':
            for graph in loader:
                graph = graph.to(device)
                out = model(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
                out = F.softmax(out)
                pred = out.argmax(dim=1)
                correct += int((pred == graph.y).sum())
            return correct / len(loader.dataset)

         elif mode == 'test':
            for graph in loader:
                graph = graph.to(device)
                out = model(graph.x, graph.edge_index,graph.edge_attr,graph.batch)
                out = F.softmax(out)
                pred = out.argmax(dim=1)
                correct += int((pred == graph.y).sum())
                Y_true = torch.cat((Y_true, graph.y), dim=0)
                Y_pred = torch.cat((Y_pred, pred), dim=0)
            #cm = confusion_matrix(Y_true, Y_pred)
            #plot_confusion_matrix(cm, ["Residential_area", "Public_area", "Industrial area", "Road", "Paddy field",
                                       #, "Forest", "Rivers", "Lake", "Coast", "Swamp", "Bare land",
                                       #"Rocky land"], "Confusion Matrix")
            Y_true = Y_true.to(CPU)
            Y_pred = Y_pred.to(CPU)
            acc = accuracy_score(Y_true, Y_pred)
            kappa = cohen_kappa_score(Y_true, Y_pred)
            print('ACC: ',  acc)
            print('kappa: ', kappa)
            return acc,kappa

    print("Start Train")


    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params}")

    best_acc = 0
    best_epoch = 0
    best_loss = 10
    print("\n==================\n")
    for epoch in range(1,max_epoch):
        loss = train()
        print('loss: ', loss)
        train_acc = test(train_loader,mode = 'train')
        test_acc = test(test_loader, mode='test1')
        if loss <= best_loss:
            best_loss = loss
            torch.save(model, PATH)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}，Test Acc: {test_acc:.4f}')

    acc,kappa = test(test_loader, mode='test')
    print(f'test_acc = {acc:.4f}')
    print(f'test_kappa = {kappa:.4f}')

    # 保存数据信息
    f = open('results.txt', 'a+')
    str_results = '\n\n=========training strat===========' \
    + " \nlearning rate=" + str(learning_rate) \
    + " epochs=" + str(max_epoch) \
    + " train_ratio=" + str(0.6) \
    + " test_ratio=" + str(0.4) \
    + " k=" + str(k) \
    + '\n======================' \
    + "\ntrain_acc=" + str(train_acc) \
    + "\ntest_acc=" + str(acc)\
    + "\nkappa=" + str(kappa)\
    + "\nDATASET=" + DATASET\
    + '\n==========training over==========='
    f.write(str_results)
    f.close()


