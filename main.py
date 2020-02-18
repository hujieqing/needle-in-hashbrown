# -*- coding: utf-8 -*-
"""Single Loading Experimenting Main Driver

The original main function did not handle arguments appropriately 
and parameter seaerch is done using a bash shell script which repeatedly
load big datasets. This main function aims to improve the experimentation process.
"""

import time
import sys
from random import shuffle
from sklearn.metrics import roc_auc_score, ndcg_score
from sklearn.metrics.pairwise import cosine_similarity
from pytorchtools import EarlyStopping
from logger import Logger
import math
from args import *
from model import *
from utils import *
from dataset import *
from losses import *
from needle_args import build_arg_paser

class TrainingData:
    '''
    The encapsulation of data list for training
    '''
    def __init__(self, data_list):
        self.data_list = data_list
        self.num_features = data_list[0].x.shape[1]
        if 'y' in data_list[0].__dict__ and data_list[0].y is not None:
            self.num_node_classes = max([data.y.max().item() for data in data_list])+1
        else:
            self.num_node_classes = None
        if 'y_graph' in data_list[0].__dict__ and data_list[0].y_graph is not None:
            self.num_graph_classes = max([data.y_graph.numpy()[0] for data in data_list])+1
        else:
            self.num_graph_classes = None

    def __str__(self):
        nodes = [data.num_nodes for data in self.data_list]
        edges = [data.num_edges for data in self.data_list]
        return '''
                1. number of graphs: {}
                2. feature dim {}
                3. number of node classes {}
                4. number of graph classes {}
                5. # of nodes: max {}, min {}, mean {}
                6. # of edges: max {}, min {}, mean {}
                '''.format(
                    len(self.data_list),
                    self.num_features,
                    self.num_node_classes,
                    self.num_graph_classes,
                    max(nodes), min(nodes), sum(nodes)/len(nodes),
                    max(edges), min(edges), sum(edges)/len(edges)
                )

def loadData(args):
    dataset_name = args.dataset
    start_time = time.time()
    data_list = get_tg_dataset(args, dataset_name, 
        use_cache=args.cache, remove_feature=args.rm_feature,
        hash_overwrite=args.hash_overwrite, hash_concat=args.hash_concat
        )
    print(dataset_name, 'load_time', time.time() - start_time)
    training_data = TrainingData(data_list)
    print(training_data)
    args.batch_size = min(args.batch_size, len(data_list))
    print('Anchor num {}, Batch size {}'.format(args.anchor_num, args.batch_size))
    return training_data

def sweep(training_data, args, parser, device, external_locals):
    locals().update(external_locals)
    '''
    sweep the following hyper parameters
    alpha, LR, WD, lambda1, lambda2
    '''
    
    logger = Logger('./logs-'+args.model+'-'+args.dataset+'-'+args.task)
    summary = [] # summary is a high-dimensinoal tuple to store HP-set v.s. average metrics from repeated training sessions
    for alpha in parser.yieldList('alpha'):
        for l1 in parser.yieldList('l1'):
            for l2 in parser.yieldList('l2'):
                if l2 == 0.0 and l1 == 0.0:
                    continue
                for lr in parser.yieldList('lr'):
                    for wd in parser.yieldList('weight_decay'):
                        results = []
                        results_ndcg = []
                        results_ktau = []
                        for repeat in range(args.repeat_num):
                            result_val = []
                            result_test = []
                            ndcg_result = []
                            ktau_result = []
                            # model
                            input_dim = training_data.num_features
                            output_dim = args.output_dim
                            data_list = training_data.data_list
                            model = None
                            if args.model == "N2V":
                                num_nodes = 0
                                for i, data in enumerate(data_list):
                                    start = num_nodes
                                    num_nodes += data.num_nodes
                                    data.x = torch.arange(start, num_nodes)

                                model = locals()[args.model](num_nodes=num_nodes, embedding_dim=args.hidden_dim,
                                                            walk_length=10, context_size=5, walks_per_node=10)
                            else:
                                model = locals()[args.model](input_dim=input_dim, feature_dim=args.feature_dim,
                                                            hidden_dim=args.hidden_dim, output_dim=output_dim,
                                                            feature_pre=args.feature_pre, layer_num=args.layer_num).to(device)

                            # data
                            for i, data in enumerate(data_list):
                                print("{0}: feature size: {1}, edge shape: {2}, edge mask train shape: {3}".format(i, data.x.shape, data.edge_index.shape, data.mask_link_positive_train.shape))
                                preselect_anchor(data, layer_num=args.layer_num, anchor_num=args.anchor_num, device=device)
                                data = data.to(device)
                                data_list[i] = data

                            # loss
                            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                            if 'link' in args.task:
                                loss_func = DistanceLoss(lambda1=l1, lambda2=l2)
                                out_act = nn.Sigmoid()

                            if args.early_stopping:
                                early_stopping = EarlyStopping(patience=50, verbose=True)

                            for epoch in range(args.epoch_num):
                                if epoch == 200:
                                    for param_group in optimizer.param_groups:
                                        param_group['lr'] /= 10
                                model.train()
                                optimizer.zero_grad()
                                shuffle(data_list)
                                effective_len = len(data_list)//args.batch_size*len(data_list)
                                for id, data in enumerate(data_list[:effective_len]):
                                    if args.permute:
                                        preselect_anchor(data, layer_num=args.layer_num, anchor_num=args.anchor_num, device=device)
                                    out = model(data)
                                    # get_link_mask(data,resplit=False)
                                    # resample negative links
                                    edge_mask_train = np.concatenate((data.mask_link_positive_train, data.mask_link_negative_train),
                                                                    axis=-1)
                                    nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[0, :]).long().to(device))
                                    nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[1, :]).long().to(device))
                                    pred = torch.sum(nodes_first * nodes_second, dim=-1)
                                    label_positive = torch.ones([data.mask_link_positive_train.shape[1], ], dtype=pred.dtype)
                                    label_negative = torch.zeros([data.mask_link_negative_train.shape[1], ], dtype=pred.dtype)
                                    label = torch.cat((label_positive, label_negative)).to(device)
                                    train_dists = 1.0 - torch.from_numpy(extract_edge_distances(data.dists_all, edge_mask_train.T, alpha)).float()
                                    normalized_pred = (1.0 - pred) / 2.0
                                    loss = loss_func(pred, label, normalized_pred, train_dists)


                                    # update
                                    loss.backward()
                                    if id % args.batch_size == args.batch_size-1:
                                        if args.batch_size > 1:
                                            # if this is slow, no need to do this normalization
                                            for p in model.parameters():
                                                if p.grad is not None:
                                                    p.grad /= args.batch_size
                                        optimizer.step()
                                        optimizer.zero_grad()

                                if epoch % args.epoch_log == 0:
                                    # evaluate
                                    model.eval()
                                    loss_train = 0
                                    loss_val = 0
                                    loss_test = 0
                                    correct_train = 0
                                    all_train = 0
                                    correct_val = 0
                                    all_val = 0
                                    correct_test = 0
                                    all_test = 0
                                    auc_train = 0
                                    auc_val = 0
                                    auc_test = 0
                                    emb_norm_min = 0
                                    emb_norm_max = 0
                                    emb_norm_mean = 0
                                    # graph distance evals (on the whole dataset)
                                    ndcg = 0
                                    ktau = 0
                                    for id, data in enumerate(data_list):
                                        out = model(data)
                                        emb_norm_min += torch.norm(out.data, dim=1).min().cpu().numpy()
                                        emb_norm_max += torch.norm(out.data, dim=1).max().cpu().numpy()
                                        emb_norm_mean += torch.norm(out.data, dim=1).mean().cpu().numpy()

                                        # train
                                        # get_link_mask(data, resplit=False)  # resample negative links
                                        edge_mask_train = np.concatenate((data.mask_link_positive_train, data.mask_link_negative_train),
                                                                        axis=-1)
                                        nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[0, :]).long()
                                                                        .to(device))
                                        nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[1, :]).long()
                                                                        .to(device))
                                        pred = torch.sum(nodes_first * nodes_second, dim=-1)
                                        label_positive = torch.ones([data.mask_link_positive_train.shape[1], ], dtype=pred.dtype)
                                        label_negative = torch.zeros([data.mask_link_negative_train.shape[1], ], dtype=pred.dtype)
                                        label = torch.cat((label_positive, label_negative)).to(device)
                                        train_dists = 1.0 - torch.from_numpy(extract_edge_distances(data.dists_all, edge_mask_train.T, alpha)).float()
                                        normalized_pred = (1.0 - pred) / 2.0
                                        loss_train += loss_func(pred, label, normalized_pred, train_dists).cpu().data.numpy()
                                        auc_train += roc_auc_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu()
                                                                .numpy())
                                        # val
                                        edge_mask_val = np.concatenate((data.mask_link_positive_val, data.mask_link_negative_val),
                                                                    axis=-1)
                                        nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_val[0, :]).long()
                                                                        .to(device))
                                        nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_val[1, :]).long()
                                                                        .to(device))
                                        pred = torch.sum(nodes_first * nodes_second, dim=-1)
                                        label_positive = torch.ones([data.mask_link_positive_val.shape[1], ], dtype=pred.dtype)
                                        label_negative = torch.zeros([data.mask_link_negative_val.shape[1], ], dtype=pred.dtype)
                                        label = torch.cat((label_positive, label_negative)).to(device)
                                        val_dists = 1.0 - torch.from_numpy(extract_edge_distances(data.dists_all, edge_mask_val.T, alpha)).float()
                                        normalized_pred = (1.0 - pred) / 2.0
                                        loss_val += loss_func(pred, label, normalized_pred, val_dists).cpu().data.numpy()
                                        auc_val += roc_auc_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu()
                                                                .numpy())
                                        # test
                                        edge_mask_test = np.concatenate((data.mask_link_positive_test, data.mask_link_negative_test),
                                                                        axis=-1)
                                        nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_test[0, :]).long()
                                                                        .to(device))
                                        nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_test[1, :]).long()
                                                                        .to(device))
                                        pred = torch.sum(nodes_first * nodes_second, dim=-1)
                                        label_positive = torch.ones([data.mask_link_positive_test.shape[1], ], dtype=pred.dtype)
                                        label_negative = torch.zeros([data.mask_link_negative_test.shape[1], ], dtype=pred.dtype)
                                        label = torch.cat((label_positive, label_negative)).to(device)
                                        test_dists = 1.0 - torch.from_numpy(extract_edge_distances(data.dists_all, edge_mask_test.T, alpha)).float()
                                        normalized_pred = (1.0 - pred) / 2.0
                                        loss_test += loss_func(pred, label, normalized_pred, test_dists).cpu().data.numpy()
                                        auc_test += roc_auc_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu()
                                                                .numpy())

                                        # evaluations for graph distance (on the while dataset)
                                        # desiderata: nodes closer to each other have higher cosine_sim
                                        # calculate pairwise cosine sim for output embeddings
                                        pairwise_sim = cosine_similarity(out.detach().cpu().numpy())
                                        # remove diagonals (similarity to itself)
                                        pairwise_sim = pairwise_sim[~np.eye(pairwise_sim.shape[0],dtype=bool)].reshape(pairwise_sim
                                                                                                                    .shape[0], -1)
                                        # low rank (closer) has higher relevance score
                                        true_relevance = 1.0 / data.dists_ranks.astype(float)
                                        # nDCG score
                                        ndcg += ndcg_score(true_relevance, pairwise_sim)
                                        # kendall's tau
                                        tau, p_value = kendalltau(true_relevance, pairwise_sim, nan_policy='omit')
                                        # # alternatively, calculates average kendall tau
                                        # tau = 0
                                        # for row in range(true_relevance.shape[0]):
                                        #     t, p = kendalltau(true_relevance[row, :], pairwise_sim[row, :])
                                        #     tau += t
                                        # tau /= true_relevance.shape[0]
                                        ktau += (0.0 if math.isnan(tau) else tau)

                                    loss_train /= id+1
                                    loss_val /= id+1
                                    loss_test /= id+1
                                    emb_norm_min /= id+1
                                    emb_norm_max /= id+1
                                    emb_norm_mean /= id+1
                                    auc_train /= id+1
                                    auc_val /= id+1
                                    auc_test /= id+1
                                    ndcg /= id+1
                                    ktau /= id+1

                                    if args.early_stopping:
                                        early_stopping(loss_val, model)

                                        if early_stopping.early_stop:
                                            print("Early stopping")
                                            break

                                    print(repeat, epoch, 'Loss {:.4f}'.format(loss_train), 'Train AUC: {:.4f}'.format(auc_train),
                                        'Val AUC: {:.4f}'.format(auc_val), 'Test AUC: {:.4f}'.format(auc_test),
                                        'nDCG: {:.4f}'.format(ndcg), 'Kendall Tau: {:.4f}'.format(ktau))


                                    result_val.append(auc_val)
                                    result_test.append(auc_test)

                                    ndcg_result.append(ndcg)
                                    ktau_result.append(ktau)

                            result_val = np.array(result_val)
                            result_test = np.array(result_test)

                            ndcg_result = np.array(ndcg_result)
                            ktau_result = np.array(ktau_result)

                            results.append(result_test[np.argmax(result_val)])
                            results_ndcg.append(np.max(ndcg_result))
                            results_ktau.append(np.nanmax(ktau_result))
                        results = np.array(results)
                        results_mean = np.mean(results).round(6)
                        results_std = np.std(results).round(6)

                        results_ndcg = np.array(results_ndcg)
                        results_ndcg_mean = np.mean(results_ndcg).round(6)
                        results_ndcg_std = np.std(results_ndcg).round(6)

                        results_ktau = np.array(results_ktau)
                        # ignoring nan values
                        results_ktau_mean = np.nanmean(results_ktau).round(6)
                        results_ktau_std = np.nanstd(results_ktau).round(6)

                        print('-----------------Final-------------------')
                        print("Parameters: alpha: {}, l1: {}, l2: {}, lr: {}, wd: {}".format(alpha, l1, l2, lr, wd))
                        print('AUC results:', results_mean, results_std)
                        print('nDCG results:', results_ndcg_mean, results_ndcg_std)
                        print('Kendall\'s Tau:', results_ktau_mean, results_ktau_std)
                        summary.append((alpha, l1, l2, lr, wd, results_mean, results_ndcg_mean, results_ktau_mean))
    with open('summary/{}_{}_{}_layer{}_approximate{}_{}.txt'.format(
        args.task,args.model,args.dataset,args.layer_num,args.approximate, args.comment
        ), 'w') as f:
        f.write('alpha,l1,l2,lr,wd,AUC,nDCG,KT\n')
        for line in summary:
            f.write(','.join([str(x) for x in line]) + '\n')
        f.close()

if __name__ == '__main__':

    if not os.path.isdir('summary'):
        os.mkdir('summary')
    # args
    parser = build_arg_paser()
    args = parser.get_args()
    print("Argument passed in are: ")
    print(args)

    if args.hash_overwrite and args.hash_concat:
        sys.exit("Do not use both hash_concat and hash_overwrite")

    # set up gpu
    if args.gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    else:
        print('Using CPU')
    device = torch.device('cuda:'+str(args.cuda) if args.gpu else 'cpu')

    data = loadData(args)
    sweep(data, args, parser, device, locals())
