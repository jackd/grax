# Pseudo-Inverse Graph Convolution Network

- [Original repository](https://github.com/dominikalfke/PinvGCN)
- [Paper](https://link.springer.com/article/10.1007/s10618-021-00752-w)

```bib
@article{alfke2021pseudoinverse,
  title={Pseudoinverse graph convolutional networks},
  author={Alfke, Dominik and Stoll, Martin},
  journal={Data Mining and Knowledge Discovery},
  pages={1--24},
  year={2021},
  publisher={Springer}
}
```

## Example Usage

```bash
python -m grax grax_config/single/fit_many.gin \
    pigcn/config/cora.gin
```

## Notes

- `PreconvolvedLinear(filters)(inputs)` (in the original repo) is just `hk.Linear(filters)(jnp.concatenate(inputs, axis=-1)`
