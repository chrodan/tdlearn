Temporal Difference Learning Algorithms for Policy Evaluation
=============================================================

What is it?
-----------
This package contains implementations of the most relevant TD methods for
policy evaluation (i.e. estimating the value function) and a benchmark framework
to systematically assess their quality in a variety of scenarios.
Only methods for linear function approximation are considered.


Implemented methods
-------------------

The following algorithms as implemented in the `td` module:

*  TD Learning with e-traces
*  TDC with e-traces
*  GTD
*  GTD2
*  LSTD with e-traces
*  Bellman Residual Minimization with or without double sampling (+ e-traces)
*  Residual Gradient with or without double sampling
*  GPTD with e-traces
*  Kalman TD
*  LSPE with e-traces
*  FPKF with e-traces

In addition, the package contains rudimentary implementations (in the `regtd` module) of different regularization
schemes for LSTD such as

*  LSTD with l2 regularization
*  Dantzig-LSTD
*  LarsTD
*  LSTD-l1
*  LSTD with l2l1 regularization


Benchmarks
----------

The package contains the implementations of several MDPs suitable for
benchmarking the algorithms (see `examples.py`). While the implementations are of more general
nature, there are ready-to-run scripts for the following benchmark scenarios:

1. 14-State Boyan Chain [`boyan`]
2. Baird Star Example [`baird`]
3. 400-State Random MDP On-policy [`disc_random_on`]
4. 400-State Random MDP Off-policy [`disc_random_off`]
5. Lin. Cart-Pole Balancing On-pol. Imp. Feat. [`lqr_imp_onpolicy`]
6. Lin. Cart-Pole Balancing Off-pol. Imp. Feat. [`lqr_imp_offpolicy`]
7. 4-dim. State Pole Balancing Onpolicy Perfect Features [`lqr_full_onpolicy`]
8. Lin. Cart-Pole Balancing Off-pol. Perf. Features [`lqr_full_offpolicy`]
9. Cart-Pole Swinup On-policy [`swingup_gauss_onpolicy`]
10. Cart-Pole Swinup Off-policy [`swingup_gauss_offpolicy`]
11. 20-link Lin. Pole Balancing On-policy [`link20_imp_onpolicy`]
12. 20-link Lin. Pole Balancing Off-policy [`link20_imp_offpolicy`]

The scripts are located in the `experiments` folder and should be executed from the base directory.
The results of the experiments is stored in the `data` folder. The `plots` directory contains scripts which automtically
create the figures of the paper *Dann, Neumann, Peters -- Policy Evaluation with Temporal Differences: A Survey and Comparison* from
the stored results. 
Alternatively, the data can be viewed interactively by executing

```python
from experiments import *
name = "lqr_full_offpolicy" # the name of the experiment (in brackets above)
measure = "RMSPBE" # Root Mean squared projected bellman error, alternatives: RMSE, RMSBE 
plot_experiment(name, measure)
```
Be aware that the scripts make heavy use of harddisk caching to avoid re-computation of runtime intensive results. This will significantly
speed-up re-executions of experiments. The cache is located
in the `cache` folder any may grow up to several GB.


Grid-search for hyper-parameter tuning
--------------------------------------

Exhaustive grid-search is implemented for tuning hyper-parameters of the algorithms. To find optimal parameters for a given benchmark 
use the script `experiments/gridsearch.py`.
The script takes the following parameters:

*  --experiment: name of the benchmark. It must be a module in the `experiments` folder. The grid-search script automatically
   imports the module to use the settings defined there
*  --njobs: number of cores to use in parallel
*  --batchsize: number of parameter settings to evaluate per job. Increasing the value may speed-up the search for small benchmarks due to the additional
   overhead per job.

For example, finding parameters for the discrete random MDP on-policy benchmark (3) can be started with

```bash
python experiments/gridsearch.py --experiment disc_random_on
```

The results of the grid-search are stored in a directory with the name of the benchmark located in the `data` folder.
You may want to have a look at 2d-slices of the hyper-parameter space. The `plot_2d_error_grid_experiment` in the experiments package
will help you. For example, the performance depencency of the FPKF on its alpha and mins parameter for fixed lambda=0 and beta=100 on the discrete random MDP on-policy benchmark can
be illustrated by

```python
from experiments import * 
plot_2d_error_grid_experiment("disc_random_on", "FPKF", criterion="RMSE", pn1="alpha", pn2="mins", settings={"beta": 100, "lam": 0.})
```

For further information how to display data have a look at the scripts in the `plots` directory.

Setup
-----
This code is known to run well with
*  Python 2.7
*  Numpy 1.6.1
*  matplotlib 1.2.0 (up-to date version required for error bars and smooth curves in plots)
*  Cython 0.17
*  mlabwrap 1.1 (http://mlabwrap.sourceforge.net/ , for executing the PILCO policy for the cart-pole swing-up task) 
*  custom joblib version available from https://github.com/chrodan/joblib 
   (to have custom hashing functions for more complex objects)

We provide short installation instruction for Unix systems in the following.

### Compiling Swing-up dynamics ###

The dynamics of the cart-pole swing-up benchmark are implemented in Python to make it really fast.
Therefore the `swingup_ode` module needs to be compiled. 

```bash
cython swingup_ode.pyx
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -o swingup_ode.so swingup_ode.c
```

You maybe need to adapt the Python include path to your settings.
Alternatively, the module can be compiled with distutils by executing in the base directory:

```bash
python setup.py build_ext --inplace
mv tdlearn/swingup_ode.so .
```
 
### Installing custom joblib version locally ###

The custom version of joblib can be installed locally in the directory so that it is used automatically by this framework
but does not interfere with code outside. This can be done by executing:

```bash
git pull  https://github.com/chrodan/joblib joblib_repo
ln -s joblib_repo/joblib joblib
```
 
