import pandas as pd
import numpy as np
from numpy import linalg
from emm import *

# Corpus
n = 1000
age = np.random.randint(15, 70, size=n) * 1.0
sex = np.random.choice([0.0, 1.0], p=[0.5, 0.5], size=n)
height = np.random.normal(160, 15, size=n)

corpus = pd.DataFrame({"age": age, "sex": sex, "height": height})

# Target
n = 5000

# Label 0
age_0 = np.random.randint(20,40) * 1.0
sex_0 = np.random.choice([0.0, 1.0], p=[0.4, 0.6], size=n)
height_0 = np.random.normal(170, 10, size=n)

# Label 1
age_1 = np.random.randint(50, 70, size=n) * 1.0
sex_1 = np.random.choice([0.0, 1.0], p=[0.6, 0.4], size=n)
height_1 = np.random.normal(150, 20, size=n)

# Real
print("\n\nExample 1, Max Entropy Weights")
funs = [
    lambda x: x.age,  # Mean
    lambda x: x.sex == 0 if not np.isnan(x.sex) else np.nan,
    lambda x: x.height,
    lambda x: (x.height - x.height.mean())**2
]

target_0 = [30, 0.4, [150, 10**2]]
target_1 = [60, 0.6, [150, 20**2]]

losses = [losses.EqualityLoss(25), losses.EqualityLoss(0.5), losses.EqualityLoss(5.3)]


regularizer = emm.EntropyRegularizer()
w, out, sol = emm.emm(corpus, funs, losses, regularizer, 1.0, verbose=True)
corpus["weight"] = w
print(corpus.head())
print(out)

# Real
print("\n\nExample 2, Boolean Weights")
funs = [
    lambda x: x.age,
    lambda x: x.sex == 0 if not np.isnan(x.sex) else np.nan,
    lambda x: x.height,
]
losses = [
    losses.LeastSquaresLoss(25),
    losses.LeastSquaresLoss(0.5),
    losses.LeastSquaresLoss(5.3),
]
regularizer = regularizers.BooleanRegularizer(5)
w, out, sol = emm.emm(df, funs, losses, regularizer, 1.0, verbose=True)
corpus["weight"] = w
print(corpus[corpus.weight > 0.1])
print(out)

# nans
print("\n\nExample 3, Missing values")
for i, j in zip(np.random.randint(50, size=25), np.random.randint(3, size=25)):
    corpus.iat[i, j] *= np.nan
# Real
funs = [
    lambda x: x.age,
    lambda x: x.sex == 0 if not np.isnan(x.sex) else np.nan,
    lambda x: x.height,
]
losses = [losses.EqualityLoss(25), losses.EqualityLoss(0.5), losses.EqualityLoss(5.3)]
regularizer = losses.EntropyRegularizer()
w, out, sol = emm.emm(corpus, funs, losses, regularizer, 1.0)
corpus["weight"] = w
print(corpus.head())
print(out)
