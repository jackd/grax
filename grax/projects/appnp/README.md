# Approximate Personalized Propagation of Neural Predictions

- [Original repo](https://github.com/benedekrozemberczki/APPNP
)
- [Paper](https://arxiv.org/abs/1810.05997)

```bib
@article{klicpera2018predict,
  title={Predict then propagate: Graph neural networks meet personalized pagerank},
  author={Klicpera, Johannes and Bojchevski, Aleksandar and G{\"u}nnemann, Stephan},
  journal={arXiv preprint arXiv:1810.05997},
  year={2018}
}
```

Note the paper and original repository use different versions of the citations datasets.

## Example Usage

```bash
# Exact cora
python -m grax \
    grax_config/single/fit_many.gin \
    appnp/config/impl/ppnp.gin \
    appnp/config/cora.gin
# Approximate cora
python -m grax \
    grax_config/single/fit_many.gin \
    appnp/config/impl/appnp.gin \
    appnp/config/cora.gin
```

```bash
python -m grax grax_config/single/fit_many.gin appnp/config/impl/appnp.gin appnp/config/cora.gin
# dt         = 5.15960 +- 1.50155
# test_acc   = 0.82110 +- 0.00676
# test_loss  = 0.56266 +- 0.00903
# train_acc  = 0.98643 +- 0.00674
# train_loss = 0.22272 +- 0.04440
# val_acc    = 0.80140 +- 0.00573
# val_loss   = 0.61753 +- 0.00814

python -m grax grax_config/single/fit_many.gin appnp/config/impl/appnp.gin appnp/config/citeseer.gin
# dt         = 7.70661 +- 1.40247
# test_acc   = 0.68850 +- 0.00537
# test_loss  = 0.95629 +- 0.00954
# train_acc  = 0.97417 +- 0.01880
# train_loss = 0.39387 +- 0.05887
# val_acc    = 0.69500 +- 0.00574
# val_loss   = 0.93566 +- 0.00708

python -m grax grax_config/single/fit_many.gin appnp/config/impl/appnp.gin appnp/config/pubmed.gin
# dt         = 9.42770 +- 1.76105
# test_acc   = 0.78350 +- 0.00396
# test_loss  = 0.55494 +- 0.00663
# train_acc  = 0.97667 +- 0.01528
# train_loss = 0.20078 +- 0.03899
# val_acc    = 0.79480 +- 0.00621
# val_loss   = 0.52583 +- 0.00655
```

## V2

`appnp/config/v2` contains configurations that make `APPNP` similar to [DAGNN](grax/projects/dagnn/README.md) without the adaptive weight factor. Results seem to be better than standard `APPNP` configurations.

```bash
python -m grax grax_config/single/fit_many.gin appnp/config/v2/cora.gin
# dt         = 2.09128 +- 0.98903
# test_acc   = 0.84400 +- 0.00725
# test_loss  = 0.69103 +- 0.01223
# train_acc  = 0.84571 +- 0.02619
# train_loss = 0.61343 +- 0.07193
# val_acc    = 0.82380 +- 0.00666
# val_loss   = 0.73087 +- 0.01179
python -m grax grax_config/single/fit_many.gin appnp/config/v2/citeseer.gin
# dt         = 3.94812 +- 1.88617
# test_acc   = 0.72960 +- 0.00664
# test_loss  = 1.32425 +- 0.00779
# train_acc  = 0.83917 +- 0.02077
# train_loss = 1.04749 +- 0.02489
# val_acc    = 0.71840 +- 0.00958
# val_loss   = 1.34182 +- 0.00783
python -m grax grax_config/single/fit_many.gin appnp/config/v2/pubmed.gin
# dt         = 4.69430 +- 1.33243
# test_acc   = 0.80340 +- 0.00341
# test_loss  = 0.51418 +- 0.00538
# train_acc  = 0.94167 +- 0.01708
# train_loss = 0.21661 +- 0.03203
# val_acc    = 0.81740 +- 0.00537
# val_loss   = 0.48976 +- 0.00488
python -m grax grax_config/single/fit_many.gin appnp/config/v2/cs.gin
# dt         = 9.29962 +- 1.12021
# test_acc   = 0.93404 +- 0.00202
# test_loss  = 0.21119 +- 0.00618
# train_acc  = 0.96367 +- 0.00888
# train_loss = 0.15154 +- 0.02158
# val_acc    = 0.92178 +- 0.00442
# val_loss   = 0.24834 +- 0.00479
python -m grax grax_config/single/fit_many.gin appnp/config/v2/physics.gin
# dt         = 7.30880 +- 0.89172
# test_acc   = 0.94910 +- 0.00114
# test_loss  = 0.15150 +- 0.00289
# train_acc  = 0.97000 +- 0.01183
# train_loss = 0.10114 +- 0.03337
# val_acc    = 0.94533 +- 0.00581
# val_loss   = 0.15111 +- 0.00932
python -m grax grax_config/single/fit_many.gin appnp/config/v2/computer.gin
# dt         = 24.22441 +- 3.92878
# test_acc   = 0.82356 +- 0.00549
# test_loss  = 0.59794 +- 0.02166
# train_acc  = 0.99300 +- 0.00458
# train_loss = 0.08975 +- 0.01159
# val_acc    = 0.87100 +- 0.00539
# val_loss   = 0.42404 +- 0.00749
python -m grax grax_config/single/fit_many.gin appnp/config/v2/photo.gin
# dt         = 10.01215 +- 1.92323
# test_acc   = 0.91834 +- 0.00371
# test_loss  = 0.39293 +- 0.01197
# train_acc  = 0.98750 +- 0.00791
# train_loss = 0.19094 +- 0.00611
# val_acc    = 0.92333 +- 0.00595
# val_loss   = 0.33551 +- 0.00415
```
