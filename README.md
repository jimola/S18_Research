# S18_Research

### Project Goal

This project contains the research I did during Spring of 2018 at Carnegie Mellon University.
We attempt to make an automatic "choice-maker" for differentially private algorithms.
The working code can be found in the Python/ directory.

### Installation

To view the experiments in the Python/ directory, make sure you have Jupyter notebooks set up.
I use my own fork of scikit-learn as well as a fork of DPComp, a repo that implements many
DP algorithms which I compare. These two repos are submodules of this project. To install them, run 

    git submodule update --init --recursive

after cloning. Then, the scikit-learn repository must be built by going to `Python/scikit-learn/`
and running:

    python setup.py build
    python setup.py install

The [dpcomp-core](https://github.com/jimola/dpcomp_core) repo can be
set up by (TODO).
