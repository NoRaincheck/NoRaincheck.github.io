---
title: Embedding Alignment
date: 2025-08-01
tags: ["Go", "ML", "embeddings", "Matrix"]
---

## Embedding Alignment

_August 2025_

**Why?**

Given the increasing use of vendored vector store solutions and dependencies on
AI for mission critical systems, it is important to determine ways to ensure
systems stay reliable when third party APIs degrade. Embedding alignment can be
an approach to re-map embeddings from one vendored system to another in the hope
that systems stay reliable at the cost of minimal degradation at inference time.
This approach requires only maintaining a linear mapping between embeddings
rather than duplicating vector stores for multiple vendors which can be
expensive in terms of ownership, processes and infrastructure.

**Method**

Embedding alignment can be framed as a Procrustes problem. The general approach
referenced in
[Creating Embeddings of Heterogeneous Relational
Datasets for Data Integration Tasks](https://dl.acm.org/doi/pdf/10.1145/3318464.3389742)
and [Word Translation Without Parallel Data](https://arxiv.org/abs/1710.04087)

The high level procedure is to determine anchors by constructing pair-wise
embeddings and then finding an appropriate (linear) mapping, typically using
[Procurstes Analysis](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.orthogonal_procrustes.html#scipy.linalg.orthogonal_procrustes).
In practise, we're interested in the _general_ class of Procrustes Problem which
devolves into solving for _(linear) homomorphism_ since there is no particular
reasons metric spaces of arbitrary embeddings reside in the same metric
space/same number of dimensions.

As an approprimation, we can greedily estimate a 'good enough' solution using
linear regression.

```py
import numpy as np
from model2vec import StaticModel
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

model_8m = StaticModel.from_pretrained("minishlab/potion-base-8M")
model_4m = StaticModel.from_pretrained("minishlab/potion-base-4M")

# define anchor corpus
corpus = [
    "It's dangerous to go alone! Take this.",
    "All your base are belong to us",
    "The cake is a lie.",
    "Hey you, you're finally awake",
    "You must construct additional pylons!"
]

X = model_4m.encode(corpus)
Y = model_8m.encode(corpus)

# determine linear transformation matrix
regr = MultiOutputRegressor(Ridge(random_state=123, fit_intercept=False)).fit(X, Y)
transformation_matrix = np.vstack([x.coef_ for x in regr.estimators_])
print(np.allclose(regr.predict(X), X.dot(transformation_matrix.T)))
```
