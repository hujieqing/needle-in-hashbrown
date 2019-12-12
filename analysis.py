import numpy as np

models_config = [['GCN', 3, -1], ['SAGE', 3, -1], ['GAT', 3, -1], ['GIN', 3, -1], ['PGNN_fast', 1, 2],
                 ['PGNN_fast', 2, 2], ['PGNN_fast', 1, -1], ['PGNN_fast', 2, -1]]

fname_missing = []
for task in ['link', 'link_pair']:
    if task == 'link':
        datasets_name = ['Cora', 'grid', 'communities', 'ppi']
    else:
        datasets_name = ['communities', 'email', 'protein']
    for dataset_name in datasets_name:
        for model_config in models_config:
            results = []
            for repeat in range(10):
                fname = 'results/{}_{}_{}_layer{}_approximate{}_repeat{}.txt'.format(
                    task, model_config[0], dataset_name, model_config[1], model_config[2], repeat)
                try:
                    with open(fname, 'r') as f:
                        result = f.read()
                        results.append(float(result))
                except:
                    fname_missing.append(fname)
            results = np.array(results)
            print('{}\t\t\t\t\t{}\t{}\t{}'.format(fname, np.mean(results).round(4), np.std(results).round(4),
                                                  len(results)))

