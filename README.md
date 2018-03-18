# Extended-OGER

![Project Status: Active – The project has reached a stable, usable
state and is being actively
developed.](http://www.repostatus.org/badges/latest/active.svg)
[![lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg)](https://www.tidyverse.org/lifecycle/#dormant)

## Introduction

Oger is a Python toolbox for rapidly building, training and evaluating hierarchical learning architectures on large datasets. It focuses mainly on Echo State Networks (ESN) and it builds functionality on top of the Modular toolkit for Data Processing (MDP). The functionality built on top of MDP includes:
 - Cross-validation of datasets
 - Grid-searching large parameter spaces
 - Processing of temporal datasets
 - Gradient-based training of deep learning architectures
 - Interface to the Speech Processing, Recognition, and Automatic Annotation Kit (SPRAAK) 

 In addition, several additional MDP nodes are provided by the Oger, such as a:
 - Reservoir node
 - Extended reservoir node with input to output mapping *
 - Leaky reservoir node
 - Ridge regression node
 - Conditional Restricted Boltzmann Machine (CRBM) node
 - Delay line reservoir (DLR) node *
 - Delay line with feedback reservoir (DLRB) node *
 - Simple cycle reservoir (SCR) node *
 - Cycle reservoir with jumps (CRJ) node *
 - Feed forward ESN (FF-ESN) reservoir node *
 - Sparse and orthogonal matrices reservoir (SORM) node *
 - Cyclic SORMs reservoir (CyclicSORM) node *

To the original Oger many new ESN models presented in the literature have been added by Tilemachos Bontzorlos. The contribution is marked with star (*) in the list above. The original Oger can be acquired from the original BitBucket repository (https://bitbucket.org/benjamin_schrauwen/organic-reservoir-computing-engine).

## Documentation

Full documentation for every function and class can be found in the source code.


## Installation
### Requirements
The required dependencies for Oger are:

    Python 2.6 or higher
    Numpy 1.1 or higher
    Scipy 0.7 or higher
    Matplotlib 0.99 or higher
    MDP 3.0 or higher

Optional packages are:

    Parallel Python if you want to run simulations in parallel
    PCSIM or Brian if you want to use spiking network nodes
    Interface to SPRAAK if you want to feed reservoir outputs into SPRAAK


### Linux
Most Linux distributions have the required dependencies listed above in their package repositories. For instance, on Ubuntu you can install the dependencies with:

    $ sudo apt-get install python-numpy python-scipy python-mdp

The latest stable version of Oger, version 1.2, can be cloned from the repository (git clone https://github.com/tilemmpon/Extended-OGER) and the included script in src folder can be run as follows:

    $ sudo python setup.py install
or in the folder of the tarball (provided here: https://github.com/tilemmpon/Extended-OGER/blob/master/Oger-1.2.tar.gz) with pip install the tarball:

    $ pip install Oger-1.2.tar.gz
I strongly reccommend to use anaconda for python development. In an anaconda environment install the tarball like above using pip.

### Mac OS/X (original instructions from Oger)
For Mac OS/X, the Scipy Superpack is a convenient way to install the latest versions of the required dependencies listed above. Note that has only been tested with the standard Python (2.6.4) included with Snow Leopard. 

Installation is identical to the installation for Linux, described above. 
### Windows

For Windows, the easiest option to install the required dependencies for Oger is to use the Python(x,y) installer. This is a scientific Python distribution which includes, a.o., the dependencies listed above but also other useful tools such as Eclipse and IPython.

Unfortunately, I have not tested how Oger can be installed in Python(x,y). If pip is available the command above should work.

## Small example usage

Bellow there is a small example (from the original source of Oger) how to use Oger. Some jupyter notebooks containing benchmarks with more examples and details can be found in the [benchmarks folder](https://github.com/tilemmpon/Extended-OGER/tree/master/benchmarks). The MDP tutorial is strongly reoccomended before you start working with Oger.

Tthis loads all methods and Oger nodes into the namespace Oger:

    import Oger
The usual experiment consists of generating the dataset, constructing your flow (learning architecture) by concatenating nodes into a feedforward graph-like structure, and then simply training and appliying it or performing some optimization or parameter sweeps. You can create a node by simpy instantiating it:

    resnode = Oger.nodes.ReservoirNode(output_dim = 100) 

This creates a reservoir node of 100 neurons. To test the newly added nodes in Extended Oger use one of the following examples instead:

    resnode = Oger.nodes.CyclicSORMsReservoir(output_dim = 100)

    resnode = Oger.nodes.SparseAndOrthogonalMatricesReservoir(output_dim = 100)

    resnode = Oger.nodes.DelayLineReservoirNode(output_dim = 100)

Let's create a ridge regression readout node:

    readoutnode = Oger.nodes.RidgeRegressionNode()

Flows can be easily created from nodes as follows:

    flow = resnode + readoutnode 

For more examples on how to construct flows, including more complex architectures, please see the MDP tutorial. Next, we can create data from one of the built-in datasets, a 30th order nonlinear autoregressive moving average system (NARMA). The input to the system is uniform white noise, the output is given by the NARMA system. 

    x,y = Oger.datasets.narma30()

By default, this returns two lists of ten 1D timeseries, the input and corresponding output, of 1000 timesteps each. Please see the API documentation for the arguments to this and other functions. 

Flows are trained in a feedforward way, node by node starting from the front. In our case, the first node (the reservoir) is not trainable, so we don't need to provide data for that. To train the second node (the readout), we provide a list of input-output tuples, conveniently generated by the zip function. We will train on the first nine timeseries, and keep the last one separate for testing later on. This gives the following dataset usable for training the flow:

    data = [None, zip(x[0:-1],y[0:-1])]

We can now train the flow to reproduce the output given the input like so:

    flow.train(data)

We can now see how our trained architecture performs on unseen data:

    plot(flow(x[-1]))
    plot(y[-1]) 
The classifier output should be similar to the original data.

## Acknowledgement

I would like to thank all the authors for the original Oger (available in https://bitbucket.org/benjamin_schrauwen/organic-reservoir-computing-engine), which I have extended. If you use this or the original software please cite it as:

    Verstraeten, D., Schrauwen, B., Dieleman, S., Brakel, P., Buteneers, P., & Pecevski, D. (2012). Oger: modular learning architectures for large-scale sequential processing. Journal of Machine Learning Research, 13(Oct), 2995-2998.

## License

Extended-OGER is licensed under GNU LESSER GENERAL PUBLIC LICENSE Version 3 like the original Oger.

## Contribution

To report an issue use the GitHub issue tracker. Please provide as much information as you can.

Contributions are always welcome. Open an issue to contact me. The preferred method of contribution is through a github pull request. Please note that there is a [code of conduct](https://github.com/tilemmpon/Extended-OGER/blob/master/code-of-conduct.md).
