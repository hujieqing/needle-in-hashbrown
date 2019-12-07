from sklearn.metrics import roc_auc_score, ndcg_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import kendalltau, spearmanr
from tensorboardX import SummaryWriter

from args import *
from model import *
from utils import *
from dataset import *

if not os.path.isdir('results'):
    os.mkdir('results')
# args
args = make_args()
print(args)
np.random.seed(123)
np.random.seed()
writer_train = SummaryWriter(comment=args.task+'_'+args.model+'_'+args.comment+'_train')
writer_val = SummaryWriter(comment=args.task+'_'+args.model+'_'+args.comment+'_val')
writer_test = SummaryWriter(comment=args.task+'_'+args.model+'_'+args.comment+'_test')

# set up gpu
if args.gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
else:
    print('Using CPU')
device = torch.device('cuda:'+str(args.cuda) if args.gpu else 'cpu')


for task in ['link', 'link_pair']:
    args.task = task
    if args.dataset=='All':
        if task == 'link':
            datasets_name = ['grid','communities','ppi']
        else:
            datasets_name = ['communities', 'email', 'protein']
    else:
        datasets_name = [args.dataset]
    for dataset_name in datasets_name:
        # if dataset_name in ['communities','grid']:
        #     args.cache = False
        # else:
        #     args.epoch_num = 401
        #     args.cache = True
        results = []
        results_ndcg = []
        results_ktau = []
        for repeat in range(args.repeat_num):
            result_val = []
            result_test = []
            ndcg_result = []
            ktau_result = []
            time1 = time.time()
            data_list = get_tg_dataset(args, dataset_name, use_cache=args.cache, remove_feature=args.rm_feature, hash_overwrite=args.hash_overwrite)
            time2 = time.time()
            print(dataset_name, 'load time',  time2-time1)

            num_features = data_list[0].x.shape[1]
            num_node_classes = None
            num_graph_classes = None
            if 'y' in data_list[0].__dict__ and data_list[0].y is not None:
                num_node_classes = max([data.y.max().item() for data in data_list])+1
            if 'y_graph' in data_list[0].__dict__ and data_list[0].y_graph is not None:
                num_graph_classes = max([data.y_graph.numpy()[0] for data in data_list])+1
            print('Dataset', dataset_name, 'Graph', len(data_list), 'Feature', num_features, 'Node Class', num_node_classes, 'Graph Class', num_graph_classes)
            nodes = [data.num_nodes for data in data_list]
            edges = [data.num_edges for data in data_list]
            print('Node: max{}, min{}, mean{}'.format(max(nodes), min(nodes), sum(nodes)/len(nodes)))
            print('Edge: max{}, min{}, mean{}'.format(max(edges), min(edges), sum(edges)/len(edges)))

            args.batch_size = min(args.batch_size, len(data_list))
            print('Anchor num {}, Batch size {}'.format(args.anchor_num, args.batch_size))

            # model
            input_dim = num_features
            output_dim = args.output_dim
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
                            feature_pre=args.feature_pre, layer_num=args.layer_num, dropout=args.dropout).to(device)

            # data
            for i,data in enumerate(data_list):
                print("{0}: feature size: {1}".format(i, data.x.shape))
                preselect_anchor(data, layer_num=args.layer_num, anchor_num=args.anchor_num, device='cpu')
                data = data.to(device)
                data_list[i] = data

            # loss
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
            if 'link' in args.task:
                loss_func = nn.BCEWithLogitsLoss()
                out_act = nn.Sigmoid()


            for epoch in range(args.epoch_num):
                if epoch==200:
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
                    # get_link_mask(data,resplit=False)  # resample negative links
                    edge_mask_train = np.concatenate((data.mask_link_positive_train, data.mask_link_negative_train), axis=-1)
                    nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[0,:]).long().to(device))
                    nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[1,:]).long().to(device))
                    pred = torch.sum(nodes_first * nodes_second, dim=-1)
                    label_positive = torch.ones([data.mask_link_positive_train.shape[1],], dtype=pred.dtype)
                    label_negative = torch.zeros([data.mask_link_negative_train.shape[1],], dtype=pred.dtype)
                    label = torch.cat((label_positive,label_negative)).to(device)
                    loss = loss_func(pred, label)

                    # update
                    loss.backward()
                    if id % args.batch_size == args.batch_size-1:
                        if args.batch_size>1:
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
                        edge_mask_train = np.concatenate((data.mask_link_positive_train, data.mask_link_negative_train), axis=-1)
                        nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[0, :]).long().to(device))
                        nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[1, :]).long().to(device))
                        pred = torch.sum(nodes_first * nodes_second, dim=-1)
                        label_positive = torch.ones([data.mask_link_positive_train.shape[1], ], dtype=pred.dtype)
                        label_negative = torch.zeros([data.mask_link_negative_train.shape[1], ], dtype=pred.dtype)
                        label = torch.cat((label_positive, label_negative)).to(device)
                        loss_train += loss_func(pred, label).cpu().data.numpy()
                        auc_train += roc_auc_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu().numpy())
                        # val
                        edge_mask_val = np.concatenate((data.mask_link_positive_val, data.mask_link_negative_val), axis=-1)
                        nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_val[0, :]).long().to(device))
                        nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_val[1, :]).long().to(device))
                        pred = torch.sum(nodes_first * nodes_second, dim=-1)
                        label_positive = torch.ones([data.mask_link_positive_val.shape[1], ], dtype=pred.dtype)
                        label_negative = torch.zeros([data.mask_link_negative_val.shape[1], ], dtype=pred.dtype)
                        label = torch.cat((label_positive, label_negative)).to(device)
                        loss_val += loss_func(pred, label).cpu().data.numpy()
                        auc_val += roc_auc_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu().numpy())
                        # test
                        edge_mask_test = np.concatenate((data.mask_link_positive_test, data.mask_link_negative_test), axis=-1)
                        nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_test[0, :]).long().to(device))
                        nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_test[1, :]).long().to(device))
                        pred = torch.sum(nodes_first * nodes_second, dim=-1)
                        label_positive = torch.ones([data.mask_link_positive_test.shape[1], ], dtype=pred.dtype)
                        label_negative = torch.zeros([data.mask_link_negative_test.shape[1], ], dtype=pred.dtype)
                        label = torch.cat((label_positive, label_negative)).to(device)
                        loss_test += loss_func(pred, label).cpu().data.numpy()
                        auc_test += roc_auc_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu().numpy())

                        # evaluations for graph distance (on the while dataset)
                        # desiderata: nodes closer to each other have higher cosine_sim
                        # calculate pairwise cosine sim for output embeddings
                        pairwise_sim = cosine_similarity(out.detach().numpy())
                        # remove diagonals (similarity to itself)
                        pairwise_sim = pairwise_sim[~np.eye(pairwise_sim.shape[0],dtype=bool)].reshape(pairwise_sim.shape[0],-1)
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
                        ktau += tau

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

                    print(repeat, epoch, 'Loss {:.4f}'.format(loss_train), 'Train AUC: {:.4f}'.format(auc_train),
                          'Val AUC: {:.4f}'.format(auc_val), 'Test AUC: {:.4f}'.format(auc_test),
                          'nDCG: {:.4f}'.format(ndcg), 'Kendall Tau: {:.4f}'.format(ktau))
                    writer_train.add_scalar('repeat_' + str(repeat) + '/auc_'+dataset_name, auc_train, epoch)
                    writer_train.add_scalar('repeat_' + str(repeat) + '/loss_'+dataset_name, loss_train, epoch)
                    writer_val.add_scalar('repeat_' + str(repeat) + '/auc_'+dataset_name, auc_val, epoch)
                    writer_train.add_scalar('repeat_' + str(repeat) + '/loss_'+dataset_name, loss_val, epoch)
                    writer_test.add_scalar('repeat_' + str(repeat) + '/auc_'+dataset_name, auc_test, epoch)
                    writer_test.add_scalar('repeat_' + str(repeat) + '/loss_'+dataset_name, loss_test, epoch)
                    writer_test.add_scalar('repeat_' + str(repeat) + '/emb_min_'+dataset_name, emb_norm_min, epoch)
                    writer_test.add_scalar('repeat_' + str(repeat) + '/emb_max_'+dataset_name, emb_norm_max, epoch)
                    writer_test.add_scalar('repeat_' + str(repeat) + '/emb_mean_'+dataset_name, emb_norm_mean, epoch)

                    writer_test.add_scalar('repeat_' + str(repeat) + '/ndcg_'+dataset_name, ndcg, epoch)
                    writer_test.add_scalar('repeat_' + str(repeat) + '/ktau_'+dataset_name, ktau, epoch)

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
            results_ktau.append(np.max(ktau_result))
        results = np.array(results)
        results_mean = np.mean(results).round(6)
        results_std = np.std(results).round(6)

        results_ndcg = np.array(results_ndcg)
        results_ndcg_mean = np.mean(results_ndcg).round(6)
        results_ndcg_std = np.std(results_ndcg).round(6)

        results_ktau = np.array(results_ktau)
        results_ktau_mean = np.mean(results_ktau).round(6)
        results_ktau_std = np.std(results_ktau).round(6)

        print('-----------------Final-------------------')
        print('AUC results:', results_mean, results_std)
        print('nDCG results:', results_ndcg_mean, results_ndcg_std)
        print('Kendall\'s Tau:', results_ktau_mean, results_ktau_std)
        with open('results/{}_{}_{}_layer{}_approximate{}.txt'.format(args.task,args.model,dataset_name,args.layer_num,args.approximate), 'w') as f:
            f.write('{}, {}, {}, {}, {}, {}\n'.format(results_mean, results_std, results_ndcg_mean, results_ndcg_std, results_ktau_mean, results_ktau_std))

# export scalar data to JSON for external processing
writer_train.export_scalars_to_json("./all_scalars.json")
writer_train.close()
writer_val.export_scalars_to_json("./all_scalars.json")
writer_val.close()
writer_test.export_scalars_to_json("./all_scalars.json")
writer_test.close()