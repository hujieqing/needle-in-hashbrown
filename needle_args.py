import argparse


class NeedleArgparser(argparse.ArgumentParser):
    '''
    A thinly wrapped ArgumentParser to easily handle the parameters
    we care
    '''

    def __init__(self, **kvargs):
        super().__init__(**kvargs)
        self.args = None

    def parse_args(self):
        self.args = super().parse_args()

    def get_args(self):
        if not self.args:
            self.parse_args()
        return self.args
    
    def yieldList(self, arg_name, cast_type=float):
        if not self.args:
            self.parse_args()
        arg_list = vars(self.args)[arg_name]
        for arg in arg_list.split():
            yield cast_type(arg)

def build_arg_paser():
    parser = NeedleArgparser(description='process training arguments')

    parser.add_argument('--comment', dest='comment', default='', type=str,
                        help='comment')
    parser.add_argument('--task', dest='task', default='link', type=str,
                        help='link; link_pair')
    parser.add_argument('--model', dest='model', default='GCN', type=str,
                        help='model class name. E.g., GCN, PGNN, SAGE, PGNN, GIN')
    parser.add_argument('--gpu', dest='gpu', default=False, type=bool,
                        help='whether use gpu')
    parser.add_argument('--cache_no', dest='cache', action='store_false',
                        help='whether use cache')
    parser.add_argument('--cpu', dest='gpu', action='store_false',
                        help='whether use cpu')
    parser.add_argument('--cuda', dest='cuda', default='0', type=str)

    # dataset
    parser.add_argument('--dataset', dest='dataset', default='ppi', type=str,
                        help='Cora; grid; communities; ppi; email')
    parser.add_argument('--remove_link_ratio', dest='remove_link_ratio', default=0.2, type=float)
    parser.add_argument('--rm_feature', dest='rm_feature', action='store_true',
                        help='whether rm_feature')
    parser.add_argument('--rm_feature_no', dest='rm_feature', action='store_false',
                        help='whether rm_feature')
    parser.add_argument('--hash_overwrite', dest='hash_overwrite', action='store_true',
                        help='overwrite features with hash')
    parser.add_argument('--hash_overwrite_no', dest='hash_overwrite', action='store_false',
                        help='do not overwrite features with hash')
    parser.add_argument('--hash_concat', dest='hash_concat', action='store_true',
                        help='concatenate features with hash vector')
    parser.add_argument('--hash_concat_no', dest='hash_concat', action='store_false',
                        help='do not concatenate features with hash vector')
    parser.add_argument('--permute', dest='permute', action='store_true',
                        help='whether permute subsets')
    parser.add_argument('--permute_no', dest='permute', action='store_false',
                        help='whether permute subsets')
    parser.add_argument('--feature_pre', dest='feature_pre', action='store_true',
                        help='whether pre transform feature')
    parser.add_argument('--feature_pre_no', dest='feature_pre', action='store_false',
                        help='whether pre transform feature')
    parser.add_argument('--dropout', dest='dropout', action='store_true',
                        help='whether dropout, default 0.5')
    parser.add_argument('--dropout_no', dest='dropout', action='store_false',
                        help='whether dropout, default 0.5')
    parser.add_argument('--approximate', dest='approximate', default=-1, type=int,
                        help='k-hop shortest path distance. -1 means exact shortest path')
    # -1, 2

    parser.add_argument('--batch_size', dest='batch_size', default=8, type=int)
    # implemented via accumulating gradient
    parser.add_argument('--layer_num', dest='layer_num', default=2, type=int)
    parser.add_argument('--feature_dim', dest='feature_dim', default=32, type=int)
    parser.add_argument('--hidden_dim', dest='hidden_dim', default=32, type=int)
    parser.add_argument('--output_dim', dest='output_dim', default=32, type=int)
    parser.add_argument('--anchor_num', dest='anchor_num', default=64, type=int)
    parser.add_argument('--normalize_adj', dest='normalize_adj', action='store_true',
                        help='whether normalize_adj')

    parser.add_argument('--lr', dest='lr', default='0.01', type=str, help='list of learning rates')
    parser.add_argument('--epoch_num', dest='epoch_num', default=2001, type=int)
    parser.add_argument('--repeat_num', dest='repeat_num', default=2, type=int) # 10
    parser.add_argument('--epoch_log', dest='epoch_log', default=10, type=int)

    parser.add_argument('--alpha', dest='alpha', default='1.0', type=str,
                        help='hyperparameter for distance normalization')
    parser.add_argument('--early_stopping', dest='early_stopping', default=True, type=bool,
                        help='hyperparameter for distance normalization')
    parser.add_argument('--weight_decay', dest='weight_decay', default='0.0001', type=str,
                        help='L2 regularization weight')

    parser.add_argument('--dist_concat', dest='dist_concat', action='store_true',
                        help='whether to concatenate the dist in PGNN instead of multiplying')
    parser.add_argument('--dist_concat_no', dest='dist_concat', action='store_false',
                        help='whether to concatenate the dist in PGNN instead of multiplying')
    
    parser.add_argument('--l1', type=str, help='list of lambda1s, deliminated by SPACE. remember to use quotes around list', default='1.0')
    
    parser.add_argument('--l2', type=str, help='list of lambda2s, deliminated by SPACE. remember to use quotes around list', default='0.0')
  
    parser.set_defaults(gpu=False, task='link', model='GCN', dataset='email',
                        cache=False, rm_feature=False, hash_overwrite=False,
                        permute=True, feature_pre=True, dropout=True,
                        approximate=-1, normalize_adj=False, hash_concat=False, early_stopping=False,
                        dist_concat=False)

   # general    
    
    return parser