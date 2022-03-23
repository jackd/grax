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
python -m grax grax_config/single/fit_many.gin dagnn/config/cora.gin
# dt         = 2.19207 +- 1.39424
# test_acc   = 0.84490 +- 0.00251
# test_loss  = 0.58862 +- 0.00751
# train_acc  = 0.87214 +- 0.02177
# train_loss = 0.49876 +- 0.06489
# val_acc    = 0.82420 +- 0.00707
# val_loss   = 0.63125 +- 0.00717
python -m grax grax_config/single/fit_many.gin dagnn/config/cora.gin dagnn/config/mods/v1.gin
# dt         = 2.30467 +- 1.12267
# test_acc   = 0.84680 +- 0.00363
# test_loss  = 0.57162 +- 0.00758
# train_acc  = 0.86643 +- 0.02348
# train_loss = 0.48653 +- 0.05426
# val_acc    = 0.82140 +- 0.00626
# val_loss   = 0.61605 +- 0.00822

python -m grax grax_config/single/fit_many.gin dagnn/config/citeseer.gin
# dt         = 4.75985 +- 1.38186
# test_acc   = 0.73090 +- 0.00532
# test_loss  = 1.12995 +- 0.00775
# train_acc  = 0.88167 +- 0.01700
# train_loss = 0.75056 +- 0.01844
# val_acc    = 0.72780 +- 0.00603
# val_loss   = 1.15377 +- 0.00373
python -m grax grax_config/single/fit_many.gin dagnn/config/citeseer.gin dagnn/config/mods/v1.gin
# dt         = 3.20875 +- 1.23792
# test_acc   = 0.72990 +- 0.00448
# test_loss  = 1.11899 +- 0.00563
# train_acc  = 0.88583 +- 0.01902
# train_loss = 0.75158 +- 0.01689
# val_acc    = 0.72960 +- 0.00662
# val_loss   = 1.14280 +- 0.00514

python -m grax grax_config/single/fit_many.gin dagnn/config/pubmed.gin
# dt         = 4.41597 +- 1.68038
# test_acc   = 0.80330 +- 0.00537
# test_loss  = 0.51997 +- 0.00860
# train_acc  = 0.95500 +- 0.01302
# train_loss = 0.19683 +- 0.03559
# val_acc    = 0.82160 +- 0.00991
# val_loss   = 0.48851 +- 0.00482
python -m grax grax_config/single/fit_many.gin dagnn/config/pubmed.gin dagnn/config/mods/v1.gin
# dt         = 3.45478 +- 1.38481
# test_acc   = 0.80810 +- 0.00423
# test_loss  = 0.52527 +- 0.00698
# train_acc  = 0.95833 +- 0.01863
# train_loss = 0.17388 +- 0.03327
# val_acc    = 0.82020 +- 0.00841
# val_loss   = 0.48745 +- 0.00325

python -m grax grax_config/single/fit_many.gin dagnn/config/cs.gin
# dt         = 9.60916 +- 2.07880
# test_acc   = 0.93001 +- 0.00339
# test_loss  = 0.22881 +- 0.01465
# train_acc  = 0.95800 +- 0.01784
# train_loss = 0.16798 +- 0.05048
# val_acc    = 0.91956 +- 0.00369
# val_loss   = 0.26209 +- 0.00743
python -m grax grax_config/single/fit_many.gin dagnn/config/cs.gin dagnn/config/mods/v1.gin
# dt         = 9.11697 +- 1.49876
# test_acc   = 0.93166 +- 0.00257
# test_loss  = 0.21914 +- 0.00788
# train_acc  = 0.96433 +- 0.01342
# train_loss = 0.14908 +- 0.03216
# val_acc    = 0.91978 +- 0.00364
# val_loss   = 0.25720 +- 0.00471

python -m grax grax_config/single/fit_many.gin dagnn/config/phyiscs.gin
# dt         = 7.68532 +- 1.35624
# test_acc   = 0.94861 +- 0.00313
# test_loss  = 0.16170 +- 0.00722
# train_acc  = 0.97500 +- 0.01432
# train_loss = 0.09185 +- 0.03391
# val_acc    = 0.94733 +- 0.00359
# val_loss   = 0.15760 +- 0.00828
python -m grax grax_config/single/fit_many.gin dagnn/config/phyiscs.gin dagnn/config/mods/v1.gin
# dt         = 7.30581 +- 1.18368
# test_acc   = 0.94901 +- 0.00119
# test_loss  = 0.15556 +- 0.00417
# train_acc  = 0.97800 +- 0.01400
# train_loss = 0.09385 +- 0.03681
# val_acc    = 0.94533 +- 0.00718
# val_loss   = 0.15259 +- 0.00907

python -m grax grax_config/single/fit_many.gin dagnn/config/computer.gin
# dt         = 27.20390 +- 6.35747
# test_acc   = 0.82849 +- 0.00258
# test_loss  = 0.60051 +- 0.01600
# train_acc  = 0.98900 +- 0.00374
# train_loss = 0.09157 +- 0.01728
# val_acc    = 0.87400 +- 0.00554
# val_loss   = 0.43147 +- 0.00407
python -m grax grax_config/single/fit_many.gin dagnn/config/computer.gin dagnn/config/mods/v1.gin
# dt         = 23.83435 +- 5.02191
# test_acc   = 0.82530 +- 0.00458
# test_loss  = 0.61072 +- 0.01472
# train_acc  = 0.99350 +- 0.00391
# train_loss = 0.08114 +- 0.01652
# val_acc    = 0.87267 +- 0.00327
# val_loss   = 0.43008 +- 0.00542

python -m grax grax_config/single/fit_many.gin dagnn/config/photo.gin
# dt         = 12.26421 +- 2.18430
# test_acc   = 0.92375 +- 0.00254
# test_loss  = 0.35012 +- 0.00548
# train_acc  = 0.98563 +- 0.00400
# train_loss = 0.17096 +- 0.00819
# val_acc    = 0.92833 +- 0.00553
# val_loss   = 0.30462 +- 0.00238
python -m grax grax_config/single/fit_many.gin dagnn/config/photo.gin dagnn/config/mods/v1.gin
# test_acc   = 0.92325 +- 0.00289
# test_loss  = 0.34823 +- 0.00691
# train_acc  = 0.99250 +- 0.00673
# train_loss = 0.15509 +- 0.00874
# val_acc    = 0.92917 +- 0.00417
# val_loss   = 0.30373 +- 0.00269
```

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
python -m grax config/cora.gin config/tune.gin
```

```txt
Best config:
dropout_rate = 0.8
num_propagations = 10
weight_decay = 0.005
Results:
test_acc   = 0.8446000397205353 +- 0.0027999903475130834
test_loss  = 0.5878929555416107 +- 0.007841244589596428
train_acc  = 0.8757142841815948 +- 0.02733017966513154
train_loss = 0.48443670868873595 +- 0.07750265734113441
val_acc    = 0.8234000325202941 +- 0.00710212364945172
val_loss   = 0.6316501200199127 +- 0.006633291546328125
```

```bash
python -m grax config/citeseer.gin config/tune.gin
```

```txt
Best config:
dropout_rate = 0.5
num_propagations = 10
weight_decay = 0.005
Results:
test_acc   = 0.7208000361919403 +- 0.006630238857744274
test_loss  = 0.8835044145584107 +- 0.009556865925348957
train_acc  = 0.9483333885669708 +- 0.018181180993092953
train_loss = 0.2646381348371506 +- 0.023522861723545434
val_acc    = 0.7298000395298004 +- 0.005399999574400435
val_loss   = 0.8953613877296448 +- 0.0066333337990236545
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
test_acc   = 0.8019000291824341 +- 0.0034190649832681053
test_loss  = 0.553540188074112 +- 0.004880022620480253
train_acc  = 0.9116667091846467 +- 0.02242270707002339
train_loss = 0.3656807512044907 +- 0.03470992547402048
val_acc    = 0.8240000426769256 +- 0.006752789126026128
val_loss   = 0.534627091884613 +- 0.0036962106435338257
```

```bash
python -m grax config/cs.gin config/tune.gin
```

```txt
Best config:
dropout_rate = 0.5
num_propagations = 5
weight_decay = 0.0005
Results:
test_acc   = 0.9305181086063385 +- 0.00249154806669832
test_loss  = 0.2667046070098877 +- 0.0071492854279390245
train_acc  = 0.9929999470710754 +- 0.004818939502563768
train_loss = 0.0885165810585022 +- 0.009171072319718938
val_acc    = 0.9242222309112549 +- 0.004494167726796264
val_loss   = 0.3007694065570831 +- 0.004205354491763253
```

```bash
python -m grax config/physics.gin config/tune.gin
```

```txt
Best config:
dropout_rate = 0.5
num_propagations = 5
weight_decay = 0.0005
Results:
test_acc   = 0.9305181086063385 +- 0.00248817111009514
test_loss  = 0.2666884630918503 +- 0.007160385428424163
train_acc  = 0.9929999470710754 +- 0.004818939502563768
train_loss = 0.08851708695292473 +- 0.009169473446080715
val_acc    = 0.9242222309112549 +- 0.004494167726796264
val_loss   = 0.3007713705301285 +- 0.004223514034132768
```

```bash
python -m grax config/computer.gin config/tune.gin
```

```txt
Best config:
dropout_rate = 0.5
num_propagations = 10
weight_decay = 0.0005
Results:
test_acc   = 0.8225914180278778 +- 0.00489509477134944
test_loss  = 0.6581727981567382 +- 0.013977507809218602
train_acc  = 0.9584999799728393 +- 0.010499989986419676
train_loss = 0.279486945271492 +- 0.016858961929667262
val_acc    = 0.8746666073799133 +- 0.004760947745297317
val_loss   = 0.5241560935974121 +- 0.005567421684345297
```

```bash
python -m grax photo.gin tune.gin
```

```txt
Best config:
dropout_rate = 0.8
num_propagations = 5
weight_decay = 5e-05
Results:
test_acc   = 0.9294906377792358 +- 0.003121584169763985
test_loss  = 0.24191309362649918 +- 0.009669407085485953
train_acc  = 0.9662500023841858 +- 0.014031217749207719
train_loss = 0.14891528636217116 +- 0.018438338065012286
val_acc    = 0.9333333969116211 +- 0.004930061784238561
val_loss   = 0.20455202013254165 +- 0.0037220910702405016
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

## IGCN Paper Results

### Baseline

```bash
python -m grax grax_config/single/fit_many.gin dagnn/config/cora.gin
# dt         = 2.18801 +- 1.27046
# test_acc   = 0.84420 +- 0.00260
# test_loss  = 0.58833 +- 0.00729
# train_acc  = 0.87571 +- 0.02733
# train_loss = 0.48676 +- 0.07794
# val_acc    = 0.82360 +- 0.00731
# val_loss   = 0.63145 +- 0.00712

python -m grax grax_config/single/fit_many.gin dagnn/config/citeseer.gin
# dt         = 4.78410 +- 1.44665
# test_acc   = 0.73070 +- 0.00553
# test_loss  = 1.13045 +- 0.00778
# train_acc  = 0.88583 +- 0.01789
# train_loss = 0.74917 +- 0.01848
# val_acc    = 0.72840 +- 0.00747
# val_loss   = 1.15409 +- 0.00347

python -m grax grax_config/single/fit_many.gin dagnn/config/pubmed.gin
# dt         = 4.54220 +- 1.69443
# test_acc   = 0.80330 +- 0.00537
# test_loss  = 0.51996 +- 0.00860
# train_acc  = 0.95500 +- 0.01302
# train_loss = 0.19683 +- 0.03559
# val_acc    = 0.82160 +- 0.00991
# val_loss   = 0.48851 +- 0.00482
```

### v1a

```bash
python -m grax grax_config/single/fit_many.gin dagnn/config/cora.gin dagnn/config/mods/v1a.gin
# dt         = 2.30721 +- 1.08314
# test_acc   = 0.84670 +- 0.00374
# test_loss  = 0.57172 +- 0.00745
# train_acc  = 0.86643 +- 0.02348
# train_loss = 0.48614 +- 0.05416
# val_acc    = 0.82140 +- 0.00607
# val_loss   = 0.61611 +- 0.00815
python -m grax grax_config/single/fit_many.gin dagnn/config/citeseer.gin dagnn/config/mods/v1a.gin
# dt         = 3.13593 +- 1.21442
# test_acc   = 0.73020 +- 0.00473
# test_loss  = 1.11929 +- 0.00500
# train_acc  = 0.88333 +- 0.01581
# train_loss = 0.75179 +- 0.01645
# val_acc    = 0.72800 +- 0.00620
# val_loss   = 1.14320 +- 0.00485
python -m grax grax_config/single/fit_many.gin dagnn/config/pubmed.gin dagnn/config/mods/v1a.gin
# dt         = 3.52146 +- 1.29896
# test_acc   = 0.80810 +- 0.00423
# test_loss  = 0.52527 +- 0.00698
# train_acc  = 0.95833 +- 0.01863
# train_loss = 0.17389 +- 0.03327
# val_acc    = 0.82020 +- 0.00841
# val_loss   = 0.48745 +- 0.00325
```

### v1b

```bash
python -m grax grax_config/single/fit_many.gin dagnn/config/cora.gin dagnn/config/mods/v1b.gin
# dt         = 6.87666 +- 3.99710
# test_acc   = 0.73850 +- 0.02250
# test_loss  = 0.88052 +- 0.02893
# train_acc  = 0.68929 +- 0.05654
# train_loss = 0.93899 +- 0.10514
# val_acc    = 0.72280 +- 0.01796
# val_loss   = 0.90483 +- 0.03005

python -m grax grax_config/single/fit_many.gin dagnn/config/citeseer.gin dagnn/config/mods/v1b.gin
# dt         = 4.96464 +- 3.50027
# test_acc   = 0.70580 +- 0.00770
# test_loss  = 0.95702 +- 0.01957
# train_acc  = 0.81583 +- 0.02311
# train_loss = 0.60594 +- 0.04327
# val_acc    = 0.70720 +- 0.01281
# val_loss   = 0.96174 +- 0.02122

python -m grax grax_config/single/fit_many.gin dagnn/config/pubmed.gin dagnn/config/mods/v1b.gin
# dt         = 33.58079 +- 6.55649
# test_acc   = 0.78120 +- 0.02004
# test_loss  = 0.61755 +- 0.02723
# train_acc  = 0.85333 +- 0.06046
# train_loss = 0.40310 +- 0.10305
# val_acc    = 0.79020 +- 0.02195
# val_loss   = 0.58249 +- 0.03497
```

### v2a

```bash
python -m grax grax_config/single/fit_many.gin dagnn/config/cora.gin dagnn/config/mods/v2a.gin
# dt         = 2.50601 +- 1.19515
# test_acc   = 0.84320 +- 0.00687
# test_loss  = 0.71094 +- 0.01303
# train_acc  = 0.84571 +- 0.03631
# train_loss = 0.63187 +- 0.06929
# val_acc    = 0.81960 +- 0.00866
# val_loss   = 0.74780 +- 0.01406
python -m grax grax_config/single/fit_many.gin dagnn/config/citeseer.gin dagnn/config/mods/v2a.gin
# dt         = 4.89166 +- 1.69713
# test_acc   = 0.72340 +- 0.00983
# test_loss  = 1.37032 +- 0.00706
# train_acc  = 0.81917 +- 0.03141
# train_loss = 1.12174 +- 0.01538
# val_acc    = 0.71480 +- 0.00538
# val_loss   = 1.38505 +- 0.00605
python -m grax grax_config/single/fit_many.gin dagnn/config/pubmed.gin dagnn/config/mods/v2a.gin
# dt         = 6.84747 +- 2.03241
# test_acc   = 0.80310 +- 0.00197
# test_loss  = 0.51510 +- 0.00664
# train_acc  = 0.93833 +- 0.01500
# train_loss = 0.23491 +- 0.02579
# val_acc    = 0.81980 +- 0.00374
# val_loss   = 0.49340 +- 0.00521
```

### v2b

```bash
python -m grax grax_config/single/fit_many.gin dagnn/config/cora.gin dagnn/config/mods/v2b.gin
# dt         = 18.03904 +- 4.61510
# test_acc   = 0.84790 +- 0.00197
# test_loss  = 0.61531 +- 0.00708
# train_acc  = 0.85214 +- 0.02047
# train_loss = 0.53546 +- 0.04897
# val_acc    = 0.82360 +- 0.00824
# val_loss   = 0.65411 +- 0.00669
python -m grax grax_config/single/fit_many.gin dagnn/config/citeseer.gin dagnn/config/mods/v2b.gin
# dt         = 19.59277 +- 6.39787
# test_acc   = 0.73290 +- 0.00709
# test_loss  = 1.19921 +- 0.00847
# train_acc  = 0.84583 +- 0.02422
# train_loss = 0.89991 +- 0.02746
# val_acc    = 0.72560 +- 0.00920
# val_loss   = 1.21974 +- 0.00750
python -m grax grax_config/single/fit_many.gin dagnn/config/pubmed.gin dagnn/config/mods/v2b.gin
# dt         = 64.08037 +- 18.10855
# test_acc   = 0.80260 +- 0.00461
# test_loss  = 0.51261 +- 0.00618
# train_acc  = 0.94000 +- 0.02603
# train_loss = 0.23562 +- 0.04109
# val_acc    = 0.81700 +- 0.00412
# val_loss   = 0.48977 +- 0.00687
```

### v3

```bash
python -m grax grax_config/single/fit_many.gin dagnn/config/cora.gin dagnn/config/mods/v3.gin
# dt         = 2.77484 +- 0.91848
# test_acc   = 0.57000 +- 0.01857
# test_loss  = 1.66217 +- 0.00674
# train_acc  = 0.61429 +- 0.03886
# train_loss = 1.58060 +- 0.02606
# val_acc    = 0.57540 +- 0.01728
# val_loss   = 1.66388 +- 0.00573
python -m grax grax_config/single/fit_many.gin dagnn/config/citeseer.gin dagnn/config/mods/v3.gin
# dt         = 1.47608 +- 0.99738
# test_acc   = 0.16260 +- 0.05815
# test_loss  = 1.79193 +- 0.00071
# train_acc  = 0.15000 +- 0.02911
# train_loss = 1.79181 +- 0.00008
# val_acc    = 0.15720 +- 0.05693
# val_loss   = 1.79228 +- 0.00064
python -m grax grax_config/single/fit_many.gin dagnn/config/pubmed.gin dagnn/config/mods/v3.gin
# dt         = 7.51280 +- 1.42970
# test_acc   = 0.78590 +- 0.00520
# test_loss  = 0.66532 +- 0.00273
# train_acc  = 0.89167 +- 0.01863
# train_loss = 0.50640 +- 0.01713
# val_acc    = 0.79660 +- 0.00390
# val_loss   = 0.65507 +- 0.00240
```

### v4

```bash
python -m grax grax_config/single/fit_many.gin dagnn/config/cora.gin dagnn/config/mods/v4.gin
# dt         = 2.82487 +- 1.08149
# test_acc   = 0.81520 +- 0.00903
# test_loss  = 1.23714 +- 0.00722
# train_acc  = 0.78714 +- 0.02162
# train_loss = 1.13560 +- 0.02807
# val_acc    = 0.78940 +- 0.00890
# val_loss   = 1.24929 +- 0.00688
python -m grax grax_config/single/fit_many.gin dagnn/config/citeseer.gin dagnn/config/mods/v4.gin
# dt         = 1.47141 +- 1.02913
# test_acc   = 0.16980 +- 0.05279
# test_loss  = 1.79206 +- 0.00142
# train_acc  = 0.15083 +- 0.02540
# train_loss = 1.79185 +- 0.00015
# val_acc    = 0.16680 +- 0.05525
# val_loss   = 1.79267 +- 0.00122
python -m grax grax_config/single/fit_many.gin dagnn/config/pubmed.gin dagnn/config/mods/v4.gin
# dt         = 7.12972 +- 1.03572
# test_acc   = 0.78650 +- 0.00652
# test_loss  = 0.62171 +- 0.00271
# train_acc  = 0.90333 +- 0.01453
# train_loss = 0.45099 +- 0.02172
# val_acc    = 0.80180 +- 0.00827
# val_loss   = 0.60690 +- 0.00163
```

### v5

```bash
python -m grax grax_config/single/fit_many.gin dagnn/config/cora.gin dagnn/config/mods/v5.gin
# dt         = 2.27502 +- 1.52734
# test_acc   = 0.84130 +- 0.00629
# test_loss  = 0.61161 +- 0.00968
# train_acc  = 0.86857 +- 0.02619
# train_loss = 0.52359 +- 0.06365
# val_acc    = 0.82360 +- 0.00758
# val_loss   = 0.65092 +- 0.01012
python -m grax grax_config/single/fit_many.gin dagnn/config/citeseer.gin dagnn/config/mods/v5.gin
# dt         = 3.98858 +- 1.44259
# test_acc   = 0.73250 +- 0.00563
# test_loss  = 1.17806 +- 0.00953
# train_acc  = 0.85833 +- 0.00833
# train_loss = 0.85543 +- 0.02823
# val_acc    = 0.73200 +- 0.01271
# val_loss   = 1.19949 +- 0.00650
python -m grax grax_config/single/fit_many.gin dagnn/config/pubmed.gin dagnn/config/mods/v5.gin
# dt         = 5.54557 +- 1.68299
# test_acc   = 0.80370 +- 0.00349
# test_loss  = 0.54696 +- 0.00715
# train_acc  = 0.95333 +- 0.02082
# train_loss = 0.19089 +- 0.02223
# val_acc    = 0.81520 +- 0.00964
# val_loss   = 0.50210 +- 0.00364
```

### v6 (v1 + v2 + v5)

```bash
python -m grax grax_config/single/fit_many.gin dagnn/config/cora.gin dagnn/config/mods/v6.gin
# dt         = 1.99471 +- 1.00545
# test_acc   = 0.84640 +- 0.00589
# test_loss  = 0.70408 +- 0.01052
# train_acc  = 0.85500 +- 0.02191
# train_loss = 0.63021 +- 0.05516
# val_acc    = 0.82400 +- 0.01088
# val_loss   = 0.74080 +- 0.01044
python -m grax grax_config/single/fit_many.gin dagnn/config/citeseer.gin dagnn/config/mods/v6.gin
# dt         = 3.87957 +- 1.25700
# test_acc   = 0.71780 +- 0.01116
# test_loss  = 1.36860 +- 0.00390
# train_acc  = 0.82000 +- 0.02478
# train_loss = 1.12863 +- 0.02033
# val_acc    = 0.70900 +- 0.01167
# val_loss   = 1.38401 +- 0.00516
python -m grax grax_config/single/fit_many.gin dagnn/config/pubmed.gin dagnn/config/mods/v6.gin
# dt         = 5.43050 +- 1.76351
# test_acc   = 0.80340 +- 0.00338
# test_loss  = 0.53165 +- 0.00470
# train_acc  = 0.94333 +- 0.02000
# train_loss = 0.23722 +- 0.03121
# val_acc    = 0.81320 +- 0.00492
# val_loss   = 0.50391 +- 0.00387
```
