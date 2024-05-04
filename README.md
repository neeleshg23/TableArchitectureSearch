# Table Architecture Search

Dependencies:
- PyTorch
- Numpy
- Cupy
- FAISS

Installation: `conda env create -f env.yaml`

CNN weights are stored at `/0_RES/1_NN/n-m.pth`.

Hyperparameter Dataset: `hyperparameters.csv`

Performance Modeling: `cd performance_model; python performance.py -s 1,1,1 -k 1,1,1`

Training a single table: 
- Write a config file in `config.json` which looks like 
```
{
    "s_subspaces": [
        3,
        3,
        3
    ],
    "k_prototypes": [
        1,
        1,
        1
    ]
}
```
- Run `python main.py -c config.json` with correct options `m` for dataset,`n` for model, and `g` for gpu.

```
python main.py -h
usage: main.py [-h] --model MODEL --dataset DATASET --config CONFIG
               [--gpu GPU]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL, -m MODEL
                        Model abbreviation
  --dataset DATASET, -d DATASET
                        Dataset name
  --config CONFIG, -c CONFIG
                        Config file name
  --gpu GPU, -g GPU     GPU number
```

## Optimization
- Run `a1_strategy_greedy` to run a greedy algorithm to optimize randomly initialized parameters
- Run `b1_strategy_qlearning` to learn an optimal policy based tabular Q-learning 
- Add data to `hyperparameters.csv` based on what was outputted from the above 
- Run `c1_costs_gradient` to take both analytical and numerical gradients with respect to the operations and storage costs

TODO: clean results directories