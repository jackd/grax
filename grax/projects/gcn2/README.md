# Simple and Deep Graph Convolutional Networks

- [Original repository](https://github.com/chennnM/GCNII)
- [Paper](https://arxiv.org/abs/2007.02133)

## Example Usage

```bash
python -m grax grax_config/single/fit_many.gin gcn2/config/cora.gin
# test_acc   = 0.85730 +- 0.00316
# test_loss  = 0.79995 +- 0.01864
# train_acc  = 0.80429 +- 0.03673
# train_loss = 0.76976 +- 0.05348
# val_acc    = 0.82560 +- 0.00709
# val_loss   = 0.83414 +- 0.01610
python -m grax grax_config/single/fit_many.gin gcn2/config/citeseer.gin
# test_acc   = 0.73010 +- 0.00969
# test_loss  = 1.23213 +- 0.02309
# train_acc  = 0.63333 +- 0.05284
# train_loss = 1.24022 +- 0.13890
# val_acc    = 0.72180 +- 0.00494
# val_loss   = 1.24865 +- 0.02137
python -m grax grax_config/single/fit_many.gin gcn2/config/pubmed.gin
# test_acc   = 0.79930 +- 0.00410
# test_loss  = 0.55250 +- 0.00742
# train_acc  = 0.93333 +- 0.05323
# train_loss = 0.21856 +- 0.10109
# val_acc    = 0.80780 +- 0.01178
# val_loss   = 0.50372 +- 0.00797

python -m grax grax_config/single/fit_many.gin \
    gcn2/config/cora.gin \
    --bindings="variant=True"
```
