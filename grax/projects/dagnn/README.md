# DAGNN

- [Paper](https://arxiv.org/abs/2007.09296)
- [Original Repository](https://github.com/mengliu1998/DeeperGNN) (pytorch)

```bibtex
@inproceedings{liu2020towards,
  title={Towards deeper graph neural networks},
  author={Liu, Meng and Gao, Hongyang and Ji, Shuiwang},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={338--348},
  year={2020}
}
```

## Examples

```bash
python -m grax grax_config/single/fit.gin config/pubmed.gin
```

```txt
{'train_acc': DeviceArray(0.9666667, dtype=float32), 'train_loss': DeviceArray(0.16117008, dtype=float32), 'val_acc': DeviceArray(0.804, dtype=float32), 'val_loss': DeviceArray(0.4904326, dtype=float32), 'test_acc': DeviceArray(0.79800004, dtype=float32), 'test_loss': DeviceArray(0.53227234, dtype=float32)}
```

```bash
python -m grax grax_config/single/fit_many.gin config/pubmed.gin
```

```txt
test_acc   = 0.8006000339984893 +- 0.0031368750447773545
test_loss  = 0.5297654092311859 +- 0.010109030183632678
train_acc  = 0.9583333790302276 +- 0.021408726068687566
train_loss = 0.19415501579642297 +- 0.04284083560716714
val_acc    = 0.8130000352859497 +- 0.006826416947298926
val_loss   = 0.49105267226696014 +- 0.007322429980614146
```

```bash
python -m grax config/pubmed.gin config/tune.gin
```

```txt
Best config:
dropout_rate = 0.8
num_propagations = 20
weight_decay = 0.02
Results:
test_acc   = 0.8032000541687012 +- 0.0043999883261879175
test_loss  = 0.5521981954574585 +- 0.0032888201041402845
train_acc  = 0.9100000381469726 +- 0.013333341479306449
train_loss = 0.34950796961784364 +- 0.012487458845848431
val_acc    = 0.8240000367164612 +- 0.007266355888776444
val_loss   = 0.5351846814155579 +- 0.004810290118452045
```

## Results

|Dataset    |Dropout|Weight Decay|Propagations|Single (%)|10-runs (%)  |Reported (%)|
|-----------|-------|------------|------------|----------|-------------|------------|
|CiteSeer   |    0.5|        2e-2|          10|     73.80|73.12 +- 0.74| 73.3 +- 0.6|
|CiteSeer*  |    0.8|            |          20|     72.80|73.23 +- 0.49|            |
|Cora       |    0.8|        5e-3|          10|     84.20|84.29 +- 0.56| 84.4 +- 0.5|
|Cora*      |    0.5|            |          10|     84.00|83.38 +- 0.61|            |
|PubMed     |    0.8|        5e-3|          20|     79.80|80.06 +- 0.31| 80.5 +- 0.5|
|PubMed*    |       |        2e-2|            |     80.90|80.19 +- 0.39|            |
|CS         |    0.8|           0|           5|     93.19|93.18 +- 0.17| 92.8 +- 0.9|
|CS*        |       |        5e-4|            |     93.42|93.17 +- 0.17|            |
|Physics    |    0.8|           0|           5|     94.31|94.08 +- 0.19| 94.0 +- 0.6|
|Physics*   |       |            |            |          |             |            |
|Computers**|    0.5|        5e-5|           5|     85.96|85.59 +- 0.42| 84.5 +- 1.2|
|Photo      |    0.5|        5e-4|           5|     92.78|92.77 +- 0.28| 92.0 +- 0.8|
|Photo*     |       |            |            |          |             |            |

*: configurations based on our own hyperparameter search. Others use hyperparameters from the original implementation.
**: Our own hyperparameter search gives results consistent with the original implementation.
