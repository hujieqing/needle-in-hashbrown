# NEEDLE + HASH for position aware graph embedding learning
## References
1. "Position-aware Graph Neural Networks".
[Jiaxuan You](https://cs.stanford.edu/~jiaxuan/), [Rex Ying](https://cs.stanford.edu/people/rexy/), [Jure Leskovec](https://cs.stanford.edu/people/jure/index.html), [Position-aware Graph Neural Networks](http://proceedings.mlr.press/v97/you19b/you19b.pdf), ICML 2019 (long oral).
2. [GraphSage](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf)
3. [node2vec](https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf)

## Installation
1. Install on company workstation with conda
   1.1 install anaconda with this [tutorial](https://docs.anaconda.com/anaconda/install/)
   1.2 create new environment with python 3.7 `conda create -n pytorch37 python=3.7`
   1.3 install pytorch in conda environment w/o GPU `conda install pytorch torchvision cpuonly -c pytorch`
   1.4 install torch-genmetrics and relevant dependencies from 3.
 
2. Install on a machine with cuda support
  2.1 follow same instruction from 1.1 - 1.2 
  2.2 `conda install pytorch torchvision cudatoolkit=10.1 -c pytorch` choose your cuda version accordingly.
  2.3 update environment variables for g++ compilers 
```Bash
$ echo $PATH
>>> /usr/local/cuda/bin:...

$ echo $CPATH
>>> /usr/local/cuda/include:...
$ echo $LD_LIBRARY_PATH
>>> /usr/local/cuda/lib64
$ echo $DYLD_LIBRARY_PATH
>>> /usr/local/cuda/lib
```
  2.4 install libraries using commands from 3. You might need to use the following two options
`--verbose --no-cache-dir`

3. Install pytorch-gemetrics
```Bash
pip install torch-scatter
pip install torch-sparse
pip install torch-cluster
pip install torch-spline-conv
pip install torch-geometric
# torch-geometric depends on networkx, so it might automatically install networkx==2.4, you can try pip install torch-geometric==1.1.2 or uninstall networkx 2.4 and install networkx 2.3)k
pip install networkx==2.3 tensorboardX matplotlib scikit-learn
```

4. Install mmh3 for supporting the hash function
```Bash
pip install mmh3
```

5. Install the right version of sklearn `pip install scikit-learn==0.22`

## References
[pytorch geometirc tutorial](https://github.com/rusty1s/pytorch_geometric)
You can see what models are already implemented there - hopefully easy to port those into the experiments


## Run
_The following commands might not work_
- 3-layer GCN, grid
```bash
python main.py --model GCN --layer_num 3 --dataset grid
```
- 2-layer P-GNN, grid
```bash
python main.py --model PGNN --layer_num 2 --dataset grid
```
- 2-layer P-GNN, grid, with 2-hop shortest path distance
```bash
python main.py --model GCN --layer_num 2 --approximate 2 --dataset grid
```
- 3-layer GCN, all datasets
```bash
python main.py --model GCN --layer_num 3 --dataset All
```
- 2-layer PGNN, all datasets
```bash
python main.py --model PGNN --layer_num 2 --dataset All
```
You are highly encouraged to tune all kinds of hyper-parameters to get better performance. We only did very limited hyper-parameter tuning.

We recommend using tensorboard to monitor the training process. To do this, you may run
```bash
tensorboard --logdir runs
```
