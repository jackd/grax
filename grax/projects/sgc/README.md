# Simplifying Graph Convolution Networks

- [Paper](https://arxiv.org/abs/1902.07153)
- [Original repo](https://github.com/Tiiiger/SGC)

```bib
@InProceedings{pmlr-v97-wu19e,
  title = {Simplifying Graph Convolutional Networks},
  author = {Wu, Felix and Souza, Amauri and Zhang, Tianyi and Fifty, Christopher and Yu, Tao and Weinberger, Kilian},
  booktitle = {Proceedings of the 36th International Conference on Machine Learning},
  pages = {6861--6871},
  year = {2019},
  publisher = {PMLR},
}
```

## Example Usage

```bash
python -m grax grax_config/single/fit_many.gin sgc/config/cora.gin
# test_acc   = 0.81040 +- 0.00066
python -m grax grax_config/single/fit_many.gin sgc/config/citeseer.gin
# test_acc   = 0.71800 +- 0.00000
python -m grax grax_config/single/fit_many.gin sgc/config/pubmed.gin
# test_acc   = 0.78970 +- 0.00046
```
