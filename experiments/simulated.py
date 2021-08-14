import pandas as pd
import numpy as np
from numpy import linalg
import emm
from emm.losses import *
from emm.reweighting import marginal

# Corpus
m = 5000
age = np.random.choice( ['0-18','18-30','30-40','40-60','60+'],
                        p=[  0.2,   0.25,   0.25,   0.2,   0.1], size=m)
sex = np.random.choice([0.0, 1.0], p=[0.5, 0.5], size=m)
height = np.random.normal(160, 15, size=m)

data = pd.DataFrame({"age": age, "sex": sex, "height": height})
dummies = pd.get_dummies(data['age'])
corpus = dummies
corpus['sex'] = data['sex']
corpus['height'] = data['height']


# Target
n = 1000



# Label 0
age_0 = np.random.choice( ['0-18','18-30','30-40','40-60','60+'],
                        p=[  0.05,   0.1,   0.25,   0.35,  0.25], size=n)
sex_0 = np.random.choice([0.0, 1.0], p=[0.6, 0.4], size=n)
height_0 = np.random.normal(170, 10, size=n)
outcome_0 = np.zeros(n)

# Label 1
age_1 = np.random.choice( ['0-18','18-30','30-40','40-60','60+'],
                        p=[  0.1,   0.1,   0.4,    0.25, 0.15], size=n)
sex_1 = np.random.choice([0.0, 1.0], p=[0.4, 0.6], size=n)
height_1 = np.random.normal(150, 10, size=n)
outcome_0 = np.ones(m)

age_bins = ['0-18','18-30','30-40','40-60','60+']
age_p_0 = [  0.05,   0.1,   0.25,   0.35,  0.25]
age_p_1 = [  0.1,   0.1,   0.4,    0.25, 0.15]
marginals = {0: [], 1:[]}
for count, ages in enumerate(age_bins):
    marginals[0] += [marginal(ages,'mean',LeastSquaresLoss(age_p_0[count]))]
    marginals[1] += [marginal(ages,'mean',LeastSquaresLoss(age_p_1[count]))]

marginals[0] += [emm.marginal('sex', 'mean', LeastSquaresLoss(0.6))]
marginals[1] += [emm.marginal('sex', 'mean', LeastSquaresLoss(0.4))]

marginals[0] += [marginal('height', 'mean', LeastSquaresLoss(170), standardize=True)]
marginals[0] += [marginal('height', 'var', LeastSquaresLoss(10**2), standardize=True)]

marginals[1] += [marginal('height', 'mean', LeastSquaresLoss(150), standardize=True)]
marginals[1] += [marginal('height', 'var', LeastSquaresLoss(10**2), standardize=True)]


regularizer = emm.EntropyRegularizer()
rwcs = emm.reweighting.generate_synth(corpus, marginals,regularizer=regularizer, lam=0.1, verbose=True)

print(emm.utils.weighted_mean(rwcs[rwcs['Outcome']==1].drop(columns=['Outcome', 'weights']), rwcs[rwcs['Outcome']==1]['weights']))



