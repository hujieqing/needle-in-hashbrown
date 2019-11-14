# NEEDLE + HASH for position aware graph embedding learning
## References
1. "Position-aware Graph Neural Networks".
[Jiaxuan You](https://cs.stanford.edu/~jiaxuan/), [Rex Ying](https://cs.stanford.edu/people/rexy/), [Jure Leskovec](https://cs.stanford.edu/people/jure/index.html), [Position-aware Graph Neural Networks](http://proceedings.mlr.press/v97/you19b/you19b.pdf), ICML 2019 (long oral).
2. GraphSage
3. Node2vec

## Installation

- Install PyTorch (tested on 1.0.0), please refer to the offical website for further details
```bash
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

pip install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter
pip install torch-sparse
pip install torch-cluster
pip install torch-spline-conv
pip install torch-geometric
# torch-geometric depends on networkx, so it might automatically install networkx==2.4, you can try pip install torch-geometric==1.1.2 or uninstall networkx 2.4 and install networkx 2.3)
pip install networkx==2.3 tensorboardX matplotlib scikit-learn
```
## References
[pytorch geometirc tutorial](https://github.com/rusty1s/pytorch_geometric)
You can see what models are already implemented there - hopefully easy to port those into the experiments


## Run
- 3-layer GCN, grid
```bash
python main.py --model GCN --num_layers 3 --dataset grid
```
- 2-layer P-GNN, grid
```bash
python main.py --model PGNN --num_layers 2 --dataset grid
```
- 2-layer P-GNN, grid, with 2-hop shortest path distance
```bash
python main.py --model GCN --num_layers 2 --approximate 2 --dataset grid
```
- 3-layer GCN, all datasets
```bash
python main.py --model GCN --num_layers 3 --dataset All
```
- 2-layer PGNN, all datasets
```bash
python main.py --model PGNN --num_layers 2 --dataset All
```
You are highly encouraged to tune all kinds of hyper-parameters to get better performance. We only did very limited hyper-parameter tuning.

We recommend using tensorboard to monitor the training process. To do this, you may run
```bash
tensorboard --logdir runs
```
