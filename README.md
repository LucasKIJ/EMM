# EMM
The aim of the library is to facilitate the seamless use of EMM with common Python data science libraries such as ```pandas``` and ```scikit-learn```. The process of generating synthetic data with EMM is divided into four parts: pre-processing, finding the optimal weights, construction of the synthetic dataset, and evaluating metrics.

### Overview
EMM is an extension to existing survey-weighting-techniques designed for generating realistic synthetic data using class-conditional summary statistics.  To create the synthetic data,  weights are allocated to samples of a large dataset thatis  representative  of  some  general  population,  referred  to  as  the corpus dataset.  The weights are chosen such that the statistics of the syntheticdata match target summary statistics, while maintaining a desirable distribution of the weights.  The method is formulated as a convex optimisation problem, balancing two objectives:  matching of target statistics andregulating  the  entropy  of  the  weights’  distribution.   Hence,  the  methodhas  been  calledEntropic  Marginal  Matching  (EMM).

At its core, this is achieved by solving the following optimisation problem:

<img src="https://github.com/LucasKIJ/EMM/blob/main/assets/readme_opti.png" alt="drawing" width="400"/>

where <img src="https://render.githubusercontent.com/render/math?math=\mathbf f"> is the summary statistics of the corpus, <img src="https://render.githubusercontent.com/render/math?math=\hat{\mathbf f}"> are the target summary statistics, and <img src="https://render.githubusercontent.com/render/math?math=\mathbf w"> is the vector of weights. The function <img src="https://render.githubusercontent.com/render/math?math=\ell(\cdot,\cdot)"> is the loss function which can be chosen by the user, the standard choice is least squares loss, given as <img src="https://render.githubusercontent.com/render/math?math=\ell(\mathbf f, \hat{\mathbf f}) = (\mathbf f - \hat{\mathbf f})^2"> . The function <img src="https://render.githubusercontent.com/render/math?math=\r(\cdot,\cdot)"> is the weight regulariser function which can be chosen by the user, the standard choice is negative entropy, given as <img src="https://render.githubusercontent.com/render/math?math=\r(\mathbf w) = \sum_{i=1}^N w_i \log w_i"> . 

### Paper
This repository contains all the code used to create the results from the Oxford MMSC Master thesis:
[Generating Synthetic Data using Entropic Marginal Matching](https://github.com/LucasKIJ/EMM/blob/main/assets/LucasJevtic_dissertation.pdf)

### Installation
Setup the Environment by installing requirements and running the ```setup.py```
- ```pip install -r requirement.txt```
- ```pip setup.py install```

### Examples
Many examples can be found in the notebooks folder.

### Citing
If you use EMM in your research, please consider citing the work using the following bibtex:
```
@misc{jevtic2021emm,
  title={Generating Sythetic Data using Entropic Marginal Matching},
  author={Lucas Jevtic},
  month={September},
  year={2021},
  howpublished={\texttt{https://github.com/LucasKIJ/EMM/blob/main/assets/LucasJevtic_dissertation.pdf}}
}
```
