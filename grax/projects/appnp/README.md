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
