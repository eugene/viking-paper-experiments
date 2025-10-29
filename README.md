# Introduction

This repository contains the code for the experiments described in the paper [VIKING: Deep variational inference with stochastic projections](https://arxiv.org/abs/2510.23684).

💡 If you're interested in building on top of the method described in the paper, we have also released [a library](https://github.com/fadel/viking) for that use case!

## Getting started

The following will install a package with most of the paper code and its dependencies.

```commandline
pip install .
```

To use an editable installation, where you can also edit the code within the `vp` module and immediately see the changes, run:

```commandline
pip install -e .
```

"Editable" means that one doesn't have to reinstall after changing the source, which is typically easier when writing Python modules.

## How to run the code

Once the package is in your Python environment, check for scripts in the `experiments/` subdirectory. They support a `--help` argument with instructions on how to run them.

# Citation

```bib
@inproceedings{fadel2025viking,
  author = {Samuel G. Fadel
            and Hrittik Roy
            and Nicholas Krämer
            and Yevgen Zainchkovskyy
            and Stas Syrota
            and Alejandro Valverde Mahou
            and Carl Henrik Ek
            and Søren Hauberg},
  title = {VIKING: Deep variational inference with stochastic projections},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {38},
  year = {2025}
}
```
