---
title: Faster AutoML Random Search
date: 2025-12-01
tags: ["ML"]
---

## Faster AutoML Random Search

_December 2025_

In today's modern world of ML it is increasingly uncommon to perform full
cross-validation when tuning models. Instead a lot of the focus (particular in
the DL space) is to make use of train/validation split with a separate holdout
with ablations.

Based on this trend, I believe when training non-DL models, we should employ the
same approach.

When doing benchmarking for this, it ends up ranging from a 2.4x to 27.8x speed
improvement for doing hyperparameter tuning.

**How does this work?**

Instead of using the `cv` parameter when doing a random/grid search, provide the
splits directly using `StratifiedShuffleSplit`, setting the `test_size`
explicitly. For example:

```py
StratifiedShuffleSplit(
    n_splits=2,
    test_size=0.2,
)
```

Then when doing say `HalvingRandomSearchCV`, the `cv` argument as the splits
specified as above.

One reciple I like doing is combining the above with some 'good' defaults for
the hyper-parameter grid search.

Generally using the parameter size suggested by TPOT has proven to be a really
good starting point without thinking too hard.

This 'lazy' approach to ML where we use good defaults is something I believe can
be done a bit better.
