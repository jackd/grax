# IGCN

## TODO

- create-cache.gin - make it work!

## Example Usage

The following examples and results from the hyperparameter search must be run from the `grax/projects/igcn/config/` directory. Alternatively, prepend non-`grax_config` gin files with `igcn/config/`.

```bash
python -m grax grax_config/single/fit_many.gin impl/ip.gin computer.gin --bindings="smooth_only=False"
python -m grax grax_config/single/fit_many.gin impl/lp.gin computer.gin --bindings="smooth_only=False"
```

```txt
dt         = 1.74865 +- 1.20948
test_acc   = 0.79869 +- 0.00857
test_loss  = 0.72518 +- 0.02491
train_acc  = 0.86600 +- 0.01921
train_loss = 0.42213 +- 0.03335
val_acc    = 0.89733 +- 0.00554
val_loss   = 0.57114 +- 0.01513
```

## Hyperparameter Search

### Cora

```bash
python -m grax impl/lp.gin cora.gin tune.gin --bindings="smooth_only=True"
```

```txt
Best config:
dropout_rate = 0.8
epsilon = 0.1
weight_decay = 0.005
Results:
test_acc   = 0.8460000336170197 +- 0.00479582073546689
test_loss  = 0.6274185419082642 +- 0.007897664766526435
train_acc  = 0.8714285671710968 +- 0.03628121064642034
train_loss = 0.5295623511075973 +- 0.0651670757709899
val_acc    = 0.825400048494339 +- 0.004737087876014694
val_loss   = 0.6638809263706207 +- 0.008136465947421845
```

```bash
python -m grax impl/lp.gin cora.gin tune.gin --bindings="smooth_only=False"
```

```txt
Best config:
dropout_rate = 0.8
epsilon = 0.1
weight_decay = 0.005
Results:
test_acc   = 0.8463000416755676 +- 0.004817683664763519
test_loss  = 0.6286221027374268 +- 0.009356379582248662
train_acc  = 0.8607142865657806 +- 0.026963699712499463
train_loss = 0.5415493220090866 +- 0.07261843352727505
val_acc    = 0.8234000444412232 +- 0.005936332327630225
val_loss   = 0.6629278898239136 +- 0.009428581428839723
```

```bash
python -m grax impl/ip.gin cora.gin tune.gin --bindings="smooth_only=True"
```

```txt
Best config:
dropout_rate = 0.8
epsilon = 0.1
weight_decay = 0.005
Results:
test_acc   = 0.8378000438213349 +- 0.005287722761023241
test_loss  = 0.8686269283294678 +- 0.010611561848047301
train_acc  = 0.700000011920929 +- 0.03319700839155919
train_loss = 0.9388962805271148 +- 0.04772475943973143
val_acc    = 0.8168000400066375 +- 0.007807688156841539
val_loss   = 0.8892502844333648 +- 0.010606614610492174
```

```bash
python -m grax impl/ip.gin cora.gin tune.gin --bindings="smooth_only=False"
```

```txt
Best config:
dropout_rate = 0.5
epsilon = 0.05
weight_decay = 0.005
Results:
test_acc   = 0.8332000434398651 +- 0.004237920934902826
test_loss  = 0.6038558423519135 +- 0.006505807580407563
train_acc  = 0.9335714399814605 +- 0.022823636427061995
train_loss = 0.3396954208612442 +- 0.03908774588240737
val_acc    = 0.8140000402927399 +- 0.00389870423535695
val_loss   = 0.6397701740264893 +- 0.007373642323614721
```

### CiteSeer

```bash
python -m grax impl/lp.gin citeseer.gin tune.gin --bindings="smooth_only=True"
```

```txt
Best config:
dropout_rate = 0.5
epsilon = 0.2
weight_decay = 0.005
Results:
test_acc   = 0.727400028705597 +- 0.005517240125515451
test_loss  = 0.9945427119731903 +- 0.0101994706473407
train_acc  = 0.9516667068004608 +- 0.014337209656068712
train_loss = 0.4462199717760086 +- 0.027653802188175654
val_acc    = 0.7308000326156616 +- 0.005670980154045556
val_loss   = 1.0177790522575378 +- 0.006487712729509406
```

```bash
python -m grax impl/lp.gin citeseer.gin tune.gin --bindings="smooth_only=False"
```

```txt
Best config:
dropout_rate = 0.8
epsilon = 0.2
weight_decay = 0.005
Results:
test_acc   = 0.7261000454425812 +- 0.004721221398933266
test_loss  = 1.0859121680259705 +- 0.010811348719978105
train_acc  = 0.7833333730697631 +- 0.03456074430786986
train_loss = 0.7934337854385376 +- 0.03969030298154843
val_acc    = 0.7296000361442566 +- 0.008236492726373308
val_loss   = 1.104728925228119 +- 0.01075581594221539
```

```bash
python -m grax impl/ip.gin citeseer.gin tune.gin --bindings="smooth_only=True"
```

```txt
Best config:
dropout_rate = 0.5
epsilon = 0.2
weight_decay = 0.005
Results:
test_acc   = 0.7066000282764435 +- 0.006545239895615768
test_loss  = 1.0441645860671998 +- 0.005740420183467553
train_acc  = 0.930833375453949 +- 0.017098576792742348
train_loss = 0.5065262496471405 +- 0.023630029926045345
val_acc    = 0.7264000296592712 +- 0.00454312118945904
val_loss   = 1.056080663204193 +- 0.005687438476606722
```

```bash
python -m grax impl/ip.gin citeseer.gin tune.gin --bindings="smooth_only=False"
```

```txt
Best config:
dropout_rate = 0.5
epsilon = 0.2
weight_decay = 0.005
Results:
test_acc   = 0.7157000243663788 +- 0.004648653568688228
test_loss  = 1.0412954688072205 +- 0.005469509702739225
train_acc  = 0.9633333861827851 +- 0.014529673902211819
train_loss = 0.4169760555028915 +- 0.028856344053159323
val_acc    = 0.7192000269889831 +- 0.008059781172946168
val_loss   = 1.0591167688369751 +- 0.004627228147501491
```

### PubMed

```bash
python -m grax impl/lp.gin pubmed.gin tune.gin --bindings="smooth_only=True"
```

```txt
Best config:
dropout_rate = 0.8
epsilon = 0.05
weight_decay = 0.005
Results:
test_acc   = 0.8061000347137451 +- 0.0034190573127234184
test_loss  = 0.5361248433589936 +- 0.006250087922579181
train_acc  = 0.9600000500679016 +- 0.015275255958431173
train_loss = 0.2093413546681404 +- 0.0324652446657128
val_acc    = 0.8206000328063965 +- 0.008765850433352138
val_loss   = 0.49676620662212373 +- 0.004690929620085584
```

```bash
python -m grax impl/lp.gin pubmed.gin tune.gin --bindings="smooth_only=False"
```

```txt
Best config:
dropout_rate = 0.8
epsilon = 0.05
weight_decay = 0.005
Results:
test_acc   = 0.8066000401973724 +- 0.0042237481537161
test_loss  = 0.5296516656875611 +- 0.0037267971520046138
train_acc  = 0.9283333718776703 +- 0.031666663132218595
train_loss = 0.24881841093301774 +- 0.04257244303957908
val_acc    = 0.8202000319957733 +- 0.006779388702499281
val_loss   = 0.4937999188899994 +- 0.004881746278989005
```

```bash
python -m grax impl/ip.gin pubmed.gin tune.gin --bindings="smooth_only=True"
```

```txt
Best config:
dropout_rate = 0.5
epsilon = 0.2
weight_decay = 0.005
Results:
test_acc   = 0.7991000473499298 +- 0.003961061531883323
test_loss  = 0.5531837821006775 +- 0.004262800634816862
train_acc  = 0.966666716337204 +- 0.02357023306403877
train_loss = 0.23349026143550872 +- 0.027197500495031886
val_acc    = 0.8216000437736511 +- 0.00714423057029426
val_loss   = 0.5311063647270202 +- 0.004682825692957539
```

```bash
python -m grax impl/ip.gin pubmed.gin tune.gin --bindings="smooth_only=False"
```

```txt
Best config:
dropout_rate = 0.5
epsilon = 0.05
weight_decay = 0.005
Results:
test_acc   = 0.8074000298976898 +- 0.005161389221717435
test_loss  = 0.5218259394168854 +- 0.006233980907479469
train_acc  = 0.9433333873748779 +- 0.02134375037483369
train_loss = 0.2041667953133583 +- 0.04513333696211518
val_acc    = 0.8172000348567963 +- 0.008109250120413757
val_loss   = 0.48490715622901914 +- 0.005156454258745644
```

### CS

```bash
python -m grax impl/lp.gin cs.gin tune.gin --bindings="smooth_only=True"
```

```txt
Best config:
dropout_rate = 0.5
epsilon = 0.2
weight_decay = 0.0005
Results:
test_acc   = 0.9304327964782715 +- 0.0020961120267429375
test_loss  = 0.28903422355651853 +- 0.008395167587788655
train_acc  = 0.9929999470710754 +- 0.0037859352866474925
train_loss = 0.1069832168519497 +- 0.005541486166439404
val_acc    = 0.928222244977951 +- 0.0029896959595828426
val_loss   = 0.3104353129863739 +- 0.0018490196901782218
```

```bash
python -m grax impl/lp.gin cs.gin tune.gin --bindings="smooth_only=False"
```

```txt
Best config:
dropout_rate = 0.8
epsilon = 0.2
weight_decay = 5e-05
Results:
test_acc   = 0.9339930713176727 +- 0.0013044105602105585
test_loss  = 0.20667098760604857 +- 0.00512892427506136
train_acc  = 0.9689999580383301 +- 0.01527889300274491
train_loss = 0.14795909747481345 +- 0.043380672687335115
val_acc    = 0.9322222471237183 +- 0.004893936631528941
val_loss   = 0.2366621360182762 +- 0.004325092782587985
```

```bash
python -m grax impl/ip.gin cs.gin tune.gin --bindings="smooth_only=True"
```

```txt
Best config:
dropout_rate = 0.5
epsilon = 0.2
weight_decay = 0.0005
Results:
test_acc   = 0.9318261861801147 +- 0.0017658289111411683
test_loss  = 0.27056177854537966 +- 0.004905553120315478
train_acc  = 0.9746666312217712 +- 0.009910703046620369
train_loss = 0.19249971807003022 +- 0.014571536845901255
val_acc    = 0.9244444608688355 +- 0.004097582181674897
val_loss   = 0.3026579529047012 +- 0.0045177811633596646
```

```bash
python -m grax impl/ip.gin cs.gin tune.gin --bindings="smooth_only=False"
```

```txt
Best config:
dropout_rate = 0.8
epsilon = 0.2
weight_decay = 0
Results:
test_acc   = 0.9336802542209626 +- 0.0021039160871703564
test_loss  = 0.21359898447990416 +- 0.0057405316820368455
train_acc  = 0.88499995470047 +- 0.018514265794765707
train_loss = 0.3437534779310226 +- 0.03969897110427304
val_acc    = 0.927111130952835 +- 0.004745369169005075
val_loss   = 0.24426132291555405 +- 0.00545288510434571
```

### Physics

```bash
python -m grax impl/lp.gin physics.gin tune.gin --bindings="smooth_only=True"
```

```txt
Best config:
dropout_rate = 0.5
epsilon = 0.05
weight_decay = 0.02
Results:
test_acc   = 0.9355342805385589 +- 0.0018212222733140662
test_loss  = 0.4015983700752258 +- 0.0070260800097968175
train_acc  = 0.9429999887943268 +- 0.011000001972379355
train_loss = 0.35426928997039797 +- 0.011038535164976982
val_acc    = 0.9553333103656769 +- 0.0042687575219315255
val_loss   = 0.35856877863407133 +- 0.0038161143223227755
```

```bash
python -m grax impl/lp.gin physics.gin tune.gin --bindings="smooth_only=False"
```

```txt
Best config:
dropout_rate = 0.8
epsilon = 0.2
weight_decay = 0.005
Results:
test_acc   = 0.9412201166152954 +- 0.0026405722388935507
test_loss  = 0.28268209993839266 +- 0.012204304276892357
train_acc  = 0.9499999761581421 +- 0.01843909072479979
train_loss = 0.27623758763074874 +- 0.03461622934296252
val_acc    = 0.9553333103656769 +- 0.004268757521931526
val_loss   = 0.24616732597351074 +- 0.006582461453131239
```

```bash
python -m grax impl/ip.gin physics.gin tune.gin --bindings="smooth_only=True"
```

```txt
Best config:
dropout_rate = 0.5
epsilon = 0.05
weight_decay = 0.02
Results:
test_acc   = 0.9399819016456604 +- 0.0019772311114445275
test_loss  = 0.4267359793186188 +- 0.012684348045113931
train_acc  = 0.8989999771118165 +- 0.012999987602233887
train_loss = 0.4853760331869125 +- 0.022952685236949075
val_acc    = 0.9646666407585144 +- 0.005206828151648089
val_loss   = 0.39849232137203217 +- 0.0037432748548481857
```

```bash
python -m grax impl/ip.gin physics.gin tune.gin --bindings="smooth_only=False"
```

```txt
Best config:
dropout_rate = 0.8
epsilon = 0.05
weight_decay = 0.005
Results:
test_acc   = 0.9409339249134063 +- 0.0018971274099902238
test_loss  = 0.293197700381279 +- 0.011206594001740518
train_acc  = 0.8229999721050263 +- 0.03257300030037334
train_loss = 0.4906070828437805 +- 0.03471878393444154
val_acc    = 0.9659999728202819 +- 0.0062893147567407316
val_loss   = 0.24312938302755355 +- 0.006798718424881579
```

### Computer

```bash
python -m grax impl/lp.gin computer.gin tune.gin --bindings="smooth_only=True"
```

```txt
Best config:
dropout_rate = 0.5
epsilon = 0.1
weight_decay = 0.0005
Results:
test_acc   = 0.8212405920028687 +- 0.003281832026668852
test_loss  = 0.6788190603256226 +- 0.010082380666539806
train_acc  = 0.9564999759197235 +- 0.016286495301010222
train_loss = 0.331397619843483 +- 0.018206911989656932
val_acc    = 0.8713332772254944 +- 0.005206828151648089
val_loss   = 0.561251413822174 +- 0.0050946551212347055
```

```bash
python -m grax impl/lp.gin computer.gin tune.gin --bindings="smooth_only=False"
```

```txt
Best config:
dropout_rate = 0.8
epsilon = 0.1
weight_decay = 0.0005
Results:
test_acc   = 0.8175374746322632 +- 0.0031173601401069304
test_loss  = 0.6727764487266541 +- 0.009612781528659495
train_acc  = 0.8634999871253968 +- 0.020006254970433714
train_loss = 0.5392371952533722 +- 0.03206947322711337
val_acc    = 0.8729999423027038 +- 0.004818939502563768
val_loss   = 0.5725223600864411 +- 0.008750943070619502
```

```bash
python -m grax impl/ip.gin computer.gin tune.gin --bindings="smooth_only=True"
```

```txt
Best config:
dropout_rate = 0.5
epsilon = 0.1
weight_decay = 0.0005
Results:
test_acc   = 0.8061486065387726 +- 0.004603089776647021
test_loss  = 0.7460772335529328 +- 0.01773199363999769
train_acc  = 0.7789999783039093 +- 0.029137602555777114
train_loss = 0.7221390426158905 +- 0.050666800921935685
val_acc    = 0.8639999508857727 +- 0.0064635669790780725
val_loss   = 0.6175972700119019 +- 0.011180940296551855
```

```bash
python -m grax impl/ip.gin computer.gin tune.gin --bindings="smooth_only=False"
```

```txt
Best config:
dropout_rate = 0.5
epsilon = 0.05
weight_decay = 5e-05
Results:
test_acc   = 0.7938747107982635 +- 0.006215972045784194
test_loss  = 0.6873573064804077 +- 0.02800141674511351
train_acc  = 0.8509999811649323 +- 0.03254228340685095
train_loss = 0.45851403176784516 +- 0.04872100960664732
val_acc    = 0.8619999527931214 +- 0.00686374688153931
val_loss   = 0.5074866890907288 +- 0.009903563056501995
```

### Photo

```bash
python -m grax impl/lp.gin photo.gin tune.gin --bindings="smooth_only=True"
```

```txt
Best config:
dropout_rate = 0.5
epsilon = 0.1
weight_decay = 5e-05
Results:
test_acc   = 0.92415691614151 +- 0.002417983383752904
test_loss  = 0.26942107677459715 +- 0.006909189074931566
train_acc  = 0.9968750059604645 +- 0.005038904438952521
train_loss = 0.05179163180291653 +- 0.01250020159746008
val_acc    = 0.932083398103714 +- 0.00528690226653292
val_loss   = 0.23109404295682906 +- 0.0032796927285795186
```

```bash
python -m grax impl/lp.gin photo.gin tune.gin --bindings="smooth_only=False"
```

```txt
Best config:
dropout_rate = 0.8
epsilon = 0.1
weight_decay = 0
Results:
test_acc   = 0.9232820808887482 +- 0.0033319835586645393
test_loss  = 0.27603204250335694 +- 0.013897660092032662
train_acc  = 0.9618750095367432 +- 0.01640550402808073
train_loss = 0.13229130059480668 +- 0.024543636437634513
val_acc    = 0.9262500584125519 +- 0.008133015163827674
val_loss   = 0.24656569808721543 +- 0.006231469831984281
```

```bash
python -m grax impl/ip.gin photo.gin tune.gin --bindings="smooth_only=True"
```

```txt
Best config:
dropout_rate = 0.5
epsilon = 0.2
weight_decay = 5e-05
Results:
test_acc   = 0.9143220126628876 +- 0.004698048052215981
test_loss  = 0.29075373113155367 +- 0.01072445226269594
train_acc  = 0.9568750023841858 +- 0.021910673889263883
train_loss = 0.15361992120742798 +- 0.044925314365961345
val_acc    = 0.9245834052562714 +- 0.003930821722962957
val_loss   = 0.2552221342921257 +- 0.005916360012249608
```

```bash
python -m grax impl/ip.gin photo.gin tune.gin --bindings="smooth_only=False"
```

```txt
Best config:
dropout_rate = 0.5
epsilon = 0.2
weight_decay = 0.0005
Results:
test_acc   = 0.8941301167011261 +- 0.005308953756380593
test_loss  = 0.4593285292387009 +- 0.00910876233908626
train_acc  = 0.959375011920929 +- 0.020775959445708394
train_loss = 0.26884104162454603 +- 0.031441924027948966
val_acc    = 0.9175000369548798 +- 0.004859144433685204
val_loss   = 0.40360272228717803 +- 0.0034642125787657033
```

## Learned IGCN

### Learned-Cora

```bash
python -m grax impl/learned-lp.gin cora.gin learned-tune.gin --bindings="smooth_only=True"
```

```txt
Best config:
dropout_rate = 0.8
weight_decay = 0.0005
Results:
test_acc   = 0.8362000405788421 +- 0.004214264384201964
test_loss  = 0.5058841109275818 +- 0.011741071622476638
train_acc  = 0.9185714304447175 +- 0.02373322002686134
train_loss = 0.27389310002326966 +- 0.054277044708665796
val_acc    = 0.8164000391960144 +- 0.0081878054007366
val_loss   = 0.5775618076324462 +- 0.015403019827988118
```

```bash
python -m grax impl/learned-lp.gin cora.gin learned-tune.gin --bindings="smooth_only=False"
```

```txt
Best config:
dropout_rate = 0.8
weight_decay = 0
Results:
test_acc   = 0.8335000455379487 +- 0.0073518699038677094
test_loss  = 0.5253022074699402 +- 0.01069120667765693
train_acc  = 0.8842857182025909 +- 0.026108086488996894
train_loss = 0.40358409881591795 +- 0.0939351759943838
val_acc    = 0.8180000245571136 +- 0.00764198759759096
val_loss   = 0.5790971577167511 +- 0.011128323879486828
```

```bash
python -m grax impl/learned-ip.gin cora.gin learned-tune.gin --bindings="smooth_only=True"
```

```txt
Best config:
dropout_rate = 0.5
weight_decay = 0.005
Results:
test_acc   = 0.8283000409603118 +- 0.003848379704526511
test_loss  = 0.5988573849201202 +- 0.008818845865879823
train_acc  = 0.9128571510314941 +- 0.03347250437470369
train_loss = 0.36592221558094024 +- 0.03188321808340533
val_acc    = 0.8152000427246093 +- 0.007440430753946644
val_loss   = 0.6346912562847138 +- 0.008579280844689865
```

```bash
python -m grax impl/learned-ip.gin cora.gin learned-tune.gin --bindings="smooth_only=False"
```

#### Learned-CiteSeer

```txt
Best config:
dropout_rate = 0.5
weight_decay = 0.005
Results:
test_acc   = 0.8344000399112701 +- 0.005063595581218812
test_loss  = 0.5895449876785278 +- 0.01266103793121126
train_acc  = 0.9300000190734863 +- 0.01491472412245089
train_loss = 0.32147274613380433 +- 0.017086267952605486
val_acc    = 0.8140000343322754 +- 0.004732860824158382
val_loss   = 0.6310314238071442 +- 0.0085006661792452
```

```bash
python -m grax impl/learned-lp.gin citeseer.gin learned-tune.gin --bindings="smooth_only=True"
```

```txt
Best config:
dropout_rate = 0.5
weight_decay = 0.005
Results:
test_acc   = 0.7176000416278839 +- 0.0051419871706125915
test_loss  = 0.8892237603664398 +- 0.009267850170017704
train_acc  = 0.9383333802223206 +- 0.01907587245925776
train_loss = 0.2925434410572052 +- 0.02688149613608487
val_acc    = 0.7218000292778015 +- 0.005015975021572111
val_loss   = 0.899663245677948 +- 0.005685294178409591
```

```bash
python -m grax impl/learned-lp.gin citeseer.gin learned-tune.gin --bindings="smooth_only=False"
```

```txt
Best config:
dropout_rate = 0.5
weight_decay = 0.005
Results:
test_acc   = 0.7198000371456146 +- 0.0034000039100908204
test_loss  = 0.8798520565032959 +- 0.008122613167673742
train_acc  = 0.9483333766460419 +- 0.023213970733556046
train_loss = 0.2643273055553436 +- 0.024423986166245817
val_acc    = 0.7196000397205353 +- 0.005425860481772553
val_loss   = 0.8909398972988128 +- 0.009315895474947104
```

```bash
python -m grax impl/learned-ip.gin citeseer.gin learned-tune.gin --bindings="smooth_only=True"
```

```txt
Best config:
dropout_rate = 0.8
weight_decay = 0.0005
Results:
test_acc   = 0.6884000301361084 +- 0.0059363289138277065
test_loss  = 0.9629001975059509 +- 0.012279424097313002
train_acc  = 0.757500046491623 +- 0.03128055033926988
train_loss = 0.6392358005046844 +- 0.061211962703440634
val_acc    = 0.7064000248908997 +- 0.011757558573176707
val_loss   = 0.9331629157066346 +- 0.015698678061859123
```

```bash
python -m grax impl/learned-ip.gin citeseer.gin learned-tune.gin --bindings="smooth_only=False"
```

```txt
Best config:
dropout_rate = 0.8
weight_decay = 0.005
Results:
test_acc   = 0.6932000339031219 +- 0.005635597735306359
test_loss  = 0.971792733669281 +- 0.005050517915083674
train_acc  = 0.70250004529953 +- 0.03801498275753097
train_loss = 0.8360437631607056 +- 0.08641745841635558
val_acc    = 0.7082000315189362 +- 0.011981653093509854
val_loss   = 0.9600888669490815 +- 0.010826848758475838
```

#### Learned-PubMed

```bash
python -m grax impl/learned-lp.gin pubmed.gin learned-tune.gin --bindings="smooth_only=True"
```

```txt
Best config:
dropout_rate = 0.5
weight_decay = 0.02
Results:
test_acc   = 0.8017000317573547 +- 0.0029681625491949433
test_loss  = 0.5753023624420166 +- 0.004032669730656681
train_acc  = 0.9116667211055756 +- 0.05678908265332682
train_loss = 0.3599534958600998 +- 0.10106890143712703
val_acc    = 0.8164000451564789 +- 0.006916650262242294
val_loss   = 0.5477026581764222 +- 0.0043937499715558555
```

```bash
python -m grax impl/learned-lp.gin pubmed.gin learned-tune.gin --bindings="smooth_only=False"
```

```txt
Best config:
dropout_rate = 0.5
weight_decay = 0.005
Results:
test_acc   = 0.8056000471115112 +- 0.004054625060223307
test_loss  = 0.5221743941307068 +- 0.004315620160110586
train_acc  = 0.991666704416275 +- 0.008333295583724976
train_loss = 0.11774125918745995 +- 0.015444265013473605
val_acc    = 0.8146000444889069 +- 0.004294179292470057
val_loss   = 0.4925521522760391 +- 0.005715073184449751
```

```bash
python -m grax impl/learned-ip.gin pubmed.gin learned-tune.gin --bindings="smooth_only=True"
```

```txt
Best config:
dropout_rate = 0.5
weight_decay = 0.005
Results:
test_acc   = 0.804900050163269 +- 0.0035902627176484818
test_loss  = 0.5250341951847076 +- 0.005729504413837971
train_acc  = 0.9483333766460419 +- 0.02522124243669658
train_loss = 0.21448957026004792 +- 0.03212230473788911
val_acc    = 0.8176000356674195 +- 0.0070313685130325996
val_loss   = 0.48464044332504275 +- 0.007349411931231121
```

```bash
python -m grax impl/learned-ip.gin pubmed.gin learned-tune.gin --bindings="smooth_only=False"
```

```txt
Best config:
dropout_rate = 0.5
weight_decay = 0.005
Results:
test_acc   = 0.8086000382900238 +- 0.003352603672058899
test_loss  = 0.5137862741947175 +- 0.00711556276501936
train_acc  = 0.9566667199134826 +- 0.023804752080606298
train_loss = 0.18188969641923905 +- 0.050633192650257106
val_acc    = 0.8192000329494477 +- 0.0066453014353262604
val_loss   = 0.48365076780319216 +- 0.0063281925270437436
```

#### Learned-CS

```bash
python -m grax impl/learned-lp.gin cs.gin learned-tune.gin --bindings="smooth_only=True"
```

```txt
Best config:
dropout_rate = 0.8
weight_decay = 5e-05
Results:
test_acc   = 0.9326508581638336 +- 0.0014385222883989373
test_loss  = 0.21666487604379653 +- 0.00593410394445507
train_acc  = 0.9669999718666077 +- 0.00874959482133264
train_loss = 0.1487535983324051 +- 0.02439192207816911
val_acc    = 0.9273333549499512 +- 0.003150555040500069
val_loss   = 0.2445322558283806 +- 0.00409408165635684
```

```bash
python -m grax impl/learned-lp.gin cs.gin learned-tune.gin --bindings="smooth_only=False"
```

```txt
Best config:
dropout_rate = 0.8
weight_decay = 0
Results:
test_acc   = 0.9338906824588775 +- 0.0028588930837483988
test_loss  = 0.208683480322361 +- 0.008612346489018702
train_acc  = 0.9453332781791687 +- 0.012578649731342881
train_loss = 0.214212167263031 +- 0.041203359554102655
val_acc    = 0.9306666791439057 +- 0.00611009770094704
val_loss   = 0.23934829533100127 +- 0.008634150571623707
```

```bash
python -m grax impl/learned-ip.gin cs.gin learned-tune.gin --bindings="smooth_only=True"
```

```txt
Best config:
dropout_rate = 0.8
weight_decay = 5e-05
Results:
test_acc   = 0.9319456160068512 +- 0.0014028665031012443
test_loss  = 0.23142396956682204 +- 0.00675370966280597
train_acc  = 0.8323332905769348 +- 0.024406283513311634
train_loss = 0.49437166154384615 +- 0.032973945948413366
val_acc    = 0.9240000247955322 +- 0.004192889351285174
val_loss   = 0.2718872666358948 +- 0.005350651633701937
```

```bash
python -m grax impl/learned-ip.gin cs.gin learned-tune.gin --bindings="smooth_only=False"
```

```txt
Best config:
dropout_rate = 0.8
weight_decay = 0
Results:
test_acc   = 0.9348461508750916 +- 0.0022129737374891293
test_loss  = 0.20826705992221833 +- 0.00883670608515604
train_acc  = 0.8743332862854004 +- 0.016736521414153948
train_loss = 0.37288649678230285 +- 0.03243069737774802
val_acc    = 0.927111130952835 +- 0.005047921587409435
val_loss   = 0.2454438865184784 +- 0.009295890196677863
```

#### Learned-Physics

```bash
python -m grax impl/learned-lp.gin physics.gin learned-tune.gin --bindings="smooth_only=True"
```

```txt
Best config:
dropout_rate = 0.8
weight_decay = 5e-05
Results:
test_acc   = 0.9326508581638336 +- 0.0014385222883989373
test_loss  = 0.21666266173124313 +- 0.005936633407642583
train_acc  = 0.9669999718666077 +- 0.00874959482133264
train_loss = 0.14875781834125518 +- 0.024390673466255054
val_acc    = 0.9273333549499512 +- 0.003150555040500069
val_loss   = 0.2445327192544937 +- 0.004093724937712636
```

```bash
python -m grax impl/learned-lp.gin physics.gin learned-tune.gin --bindings="smooth_only=False"
```

```txt
Best config:
dropout_rate = 0.8
weight_decay = 0.005
Results:
test_acc   = 0.9405367553234101 +- 0.003693634575950308
test_loss  = 0.2131863221526146 +- 0.01773540481583252
train_acc  = 0.9429999709129333 +- 0.04075536573223489
train_loss = 0.20204095169901848 +- 0.08326345833749739
val_acc    = 0.9546666324138642 +- 0.009333348274240474
val_loss   = 0.1573011502623558 +- 0.011476125939474125
```

```bash
python -m grax impl/learned-ip.gin physics.gin learned-tune.gin --bindings="smooth_only=True"
```

```txt
Best config:
dropout_rate = 0.5
weight_decay = 0.02
Results:
test_acc   = 0.9161872565746307 +- 0.013267907674576605
test_loss  = 0.3122933030128479 +- 0.01572919378157304
train_acc  = 0.9009999811649323 +- 0.02547548195268444
train_loss = 0.43583851456642153 +- 0.01498189879730861
val_acc    = 0.95666663646698 +- 0.008563498779872105
val_loss   = 0.293861648440361 +- 0.00430563441464688
```

```bash
python -m grax impl/learned-ip.gin physics.gin learned-tune.gin --bindings="smooth_only=False"
```

```txt
Best config:
dropout_rate = 0.8
weight_decay = 0.005
Results:
test_acc   = 0.9443068683147431 +- 0.0013975059470336404
test_loss  = 0.220034059882164 +- 0.008904506617234707
train_acc  = 0.8519999861717225 +- 0.028213470826965095
train_loss = 0.44421985149383547 +- 0.0509408437959477
val_acc    = 0.9613333106040954 +- 0.0049888717579351875
val_loss   = 0.2088836446404457 +- 0.004393336142646997
```

#### Learned-Computer

```bash
python -m grax impl/learned-lp.gin computer.gin learned-tune.gin --bindings="smooth_only=True"
```

```txt
Best config:
dropout_rate = 0.8
weight_decay = 0
Results:
test_acc   = 0.804898715019226 +- 0.011105123919278313
test_loss  = 0.6735775947570801 +- 0.026214605850723563
train_acc  = 0.884499979019165 +- 0.030120583808076553
train_loss = 0.37926350235939027 +- 0.05573742419763748
val_acc    = 0.8723332762718201 +- 0.007608467551001875
val_loss   = 0.5025696873664856 +- 0.010103585709013792
```

```bash
python -m grax impl/learned-lp.gin computer.gin learned-tune.gin --bindings="smooth_only=False"
```

```txt
Best config:
dropout_rate = 0.8
weight_decay = 5e-05
Results:
test_acc   = 0.8025386452674865 +- 0.00698014082123416
test_loss  = 0.6806254088878632 +- 0.03323673068089708
train_acc  = 0.8789999842643738 +- 0.027549960374203077
train_loss = 0.39198520183563235 +- 0.05530556281729379
val_acc    = 0.8713332772254944 +- 0.008844324339668053
val_loss   = 0.5104268312454223 +- 0.011356018298763955
```

```bash
python -m grax impl/learned-ip.gin computer.gin learned-tune.gin --bindings="smooth_only=True"
```

```txt
Best config:
dropout_rate = 0.5
weight_decay = 5e-05
Results:
test_acc   = 0.823864632844925 +- 0.004803611987755453
test_loss  = 0.5647626459598541 +- 0.01425803297034258
train_acc  = 0.9114999830722809 +- 0.02588918307795702
train_loss = 0.30480011701583865 +- 0.053383463042382916
val_acc    = 0.8643332839012146 +- 0.005174719963751111
val_loss   = 0.41615076661109923 +- 0.008121719253078474
```

```bash
python -m grax impl/learned-ip.gin computer.gin learned-tune.gin --bindings="smooth_only=False"
```

```txt
Best config:
dropout_rate = 0.5
weight_decay = 0.0005
Results:
test_acc   = 0.8164661228656769 +- 0.00960178067340423
test_loss  = 0.6629272818565368 +- 0.031121535340624413
train_acc  = 0.8654999911785126 +- 0.020426702516918676
train_loss = 0.5156490743160248 +- 0.02665372971515809
val_acc    = 0.8639999508857727 +- 0.004422162169834027
val_loss   = 0.550403094291687 +- 0.004805677180053064
```

#### Learned-Photo

```bash
python -m grax impl/learned-lp.gin photo.gin learned-tune.gin --bindings="smooth_only=True"
```

```txt
Best config:
dropout_rate = 0.5
weight_decay = 5e-05
Results:
test_acc   = 0.9224072456359863 +- 0.003000192299293779
test_loss  = 0.26936763525009155 +- 0.011251157426026052
train_acc  = 0.995000010728836 +- 0.0061237233836423455
train_loss = 0.041576370038092135 +- 0.008422789012339053
val_acc    = 0.928750067949295 +- 0.003930821722962957
val_loss   = 0.22559773176908493 +- 0.002918463889437959
```

```bash
python -m grax impl/learned-lp.gin photo.gin learned-tune.gin --bindings="smooth_only=False"
```

```txt
Best config:
dropout_rate = 0.8
weight_decay = 5e-05
Results:
test_acc   = 0.9141526937484741 +- 0.006158701949680876
test_loss  = 0.3027842968702316 +- 0.021908457935883037
train_acc  = 0.9406250238418579 +- 0.02240430941351943
train_loss = 0.18478861153125764 +- 0.04329897914203592
val_acc    = 0.9175000488758087 +- 0.01017214439390743
val_loss   = 0.26044038981199263 +- 0.00826674352672366
```

```bash
python -m grax impl/learned-ip.gin photo.gin learned-tune.gin --bindings="smooth_only=True"
```

```txt
Best config:
dropout_rate = 0.5
weight_decay = 0
Results:
test_acc   = 0.9045153200626374 +- 0.0034279965950556345
test_loss  = 0.31741486489772797 +- 0.011516261379552098
train_acc  = 0.9681250095367432 +- 0.011675973639852915
train_loss = 0.11269466131925583 +- 0.031144024321763216
val_acc    = 0.9266667306423187 +- 0.0050000071525751364
val_loss   = 0.2605080157518387 +- 0.004089901115555407
```

```bash
python -m grax impl/learned-ip.gin photo.gin learned-tune.gin --bindings="smooth_only=False"
```

```txt
Best config:
dropout_rate = 0.5
weight_decay = 5e-05
Results:
test_acc   = 0.8965006649494172 +- 0.00839375968224113
test_loss  = 0.339504086971283 +- 0.02424875259935692
train_acc  = 0.9775000095367432 +- 0.009354141714642248
train_loss = 0.09820878505706787 +- 0.019166947135302367
val_acc    = 0.9191667139530182 +- 0.007021809539359082
val_loss   = 0.2764336794614792 +- 0.004609788862330902
```
