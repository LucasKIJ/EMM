# EMM
The aim of the library is to facilitate the seamless use of EMM with common Python data science libraries such as ```pandas``` and ```scikit-learn```. The process of generating synthetic data with EMM is divided into four parts: pre-processing, finding the optimal weights, construction of the synthetic dataset, and evaluating metrics.

### Overview
EMM is an extension to existing survey-weighting-techniques designed for generating realistic synthetic data using class-conditional summary statistics.  To create the synthetic data,  weights are allocated to samples of a large dataset thatis  representative  of  some  general  population,  referred  to  as  the corpus dataset.  The weights are chosen such that the statistics of the syntheticdata match target summary statistics, while maintaining a desirable distribution of the weights.  The method is formulated as a convex optimisation problem, balancing two objectives:  matching of target statistics andregulating  the  entropy  of  the  weights’  distribution.   Hence,  the  methodhas  been  calledEntropic  Marginal  Matching  (EMM).

At its core, this is achieved by solving the following optimisation problem:
```math
\begin{align}
  \min_{\mathbf w} &\quad \ell(\mathbf f, \hat{\mathbf f}) + \lambda r(w) \\
  s.t. & \quad \sum_{i=1}^N w_i = 1 \\
       & \quad w_i \geq 0, \; \; \forall i 
\end{align}
```
where $f$ is the summary statistics of the corpus, $\hat{\mathbf f}$ are the target summary statistics, and $\mathbf w$ is the vector of weights.



### Pre-processing
Pre-processing includes procedures that involve transforming raw data into a format that is ready to be used in the optimisation stage.  Processes such as data importation, conversion, standardisation, one-hot encoding of categorical variables, and handling of missing data are needed to create a corpus dataset that can be used by the EMM optimisation function.
