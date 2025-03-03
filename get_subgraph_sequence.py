from HSI_dataloader import Mydataset
from torch_geometric.data import DataLoader
from module.utils import *
from module.utils.reorganizer import relabel_graph, filter_correct_data, filter_correct_data_batch
from module.utils.parser import parse_args
from rc_explainer_pool import RC_Explainer, RC_Explainer_pro, RC_Explainer_Batch,  RC_Explainer_Batch_star
from train_test_pool_batch3 import train_policy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
if __name__ == '__main__':
    for zz in [256]:
        set_seed(19930819)
        args = parse_args()
        dataset_name = 'HSC'
        _hidden_size = 256
        _num_labels = 12
        debias_flag = False
        topN = None
        batch_size = 1
        scope = 'part'
        K = zz
        path = 'D:\PhD\project\\featureattr\\reinforced_causal_explainer-master\model\HSC_'+str(K)+'_GCN.pt'
        #path = 'D:\PhD\science\project\\featureattr\\reinforced_causal_explainer-master\OHS_64_GCN.pt'
        #train_dataset = Mydataset('Data\OHS', mode = 'training', k1 = 64)
        #test_dataset = Mydataset('Data\OHS', mode = 'testing', k1 = 64)
        train_dataset = Mydataset('Data\HSC', mode = 'training', k1 = K)
        test_dataset = Mydataset('Data\HSC', mode = 'testing', k1 = K)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = torch.load(path)
        model.eval()
        model_1 = torch.load(path)
        model_1.eval()

        # refine the datasets and data loaders
        #train_dataset, train_loader = filter_correct_data_batch(model, train_dataset, train_loader, 'training', batch_size = batch_size)
        #test_dataset, test_loader = filter_correct_data_batch(model, test_dataset, test_loader, 'testing', batch_size = 1)

        rc_explainer = RC_Explainer_Batch_star(_model=model_1, _num_labels=_num_labels,
                                               _hidden_size=_hidden_size, _use_edge_attr=False).to(device)
        pro_flag = False
        lr = 1e-4
        weight_decay = 1e-5
        reward_mode = 'binary'#mutual_info,binary,cross_entropy
        optimizer = rc_explainer.get_optimizer(lr=lr, weight_decay=weight_decay, scope=scope)
        topK_ratio = 3
        save_model_path = str(dataset_name) + "_" + str(lr) + "_" + str(weight_decay) +str(K)+ "v3.pt"
        rc_explainer, best_acc_auc, best_acc_curve, best_pre, best_rec = \
            train_policy(rc_explainer, model, train_loader, test_loader, optimizer, topK_ratio,
                         debias_flag=debias_flag, topN=topN, batch_size=batch_size, reward_mode=reward_mode,
                         save_model_path=save_model_path)

        out_file = open('test.txt','a+')
        str_results = '\n\n=========explainer results===========' \
                      + " \nbest_acc_auc=" + str(best_acc_auc) \
                      + " topK_ratio=" + str(topK_ratio) \
                      + " lr=" + str(lr) \
                      + " test_ratio=" + str(0.4) \
                      + " weight_decay=" + str(weight_decay) \
                      + '\n========== over ===========' \
            # + '\ntrain time:' + str(time_train_end - time_train_start) \
        # + '\ntest time:' + str(time_test_end - time_test_start) \
        out_file.write(str_results)
        out_file.close()